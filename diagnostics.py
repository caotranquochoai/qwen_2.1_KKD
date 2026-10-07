"""Inference phase timings and background GPU telemetry for the UI and API."""

import csv
import io
import os
import subprocess
import threading
import time
import uuid

import torch

try:
    import psutil
except ImportError:
    psutil = None

_latest = None
_latest_lock = threading.Lock()


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _normalized_uuid(value):
    if isinstance(value, bytes) and len(value) == 16:
        value = uuid.UUID(bytes=value)
    return str(value or "").lower().removeprefix("gpu-").replace("-", "")


def _environment(memory_mode):
    import diffusers

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return {
        "gpu": props.name, "vram_total_gib": round(props.total_memory / 2**30, 2),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "diffusers": diffusers.__version__, "memory_mode": memory_mode,
        "gguf_cuda_kernels_requested": os.getenv("DIFFUSERS_GGUF_CUDA_KERNELS", "false"),
        "device": device,
    }


class InferenceDiagnostics:
    def __init__(self, pipeline, memory_mode):
        self.pipe = pipeline
        self.environment = _environment(memory_mode)
        self.phases = {}
        self.step_seconds = []
        self.samples = []
        self.telemetry_error = None
        self.stage = "Khởi tạo chẩn đoán"
        self.running = False
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._originals = []
        self._denoise_start = None
        self._last_step = None
        self._snapshot = {}
        props = torch.cuda.get_device_properties(self.environment["device"])
        self._gpu_uuid = _normalized_uuid(getattr(props, "uuid", ""))

    def _select_gpu(self, rows):
        if self._gpu_uuid:
            matches = [row for row in rows if _normalized_uuid(row[1]) == self._gpu_uuid]
            if len(matches) == 1:
                return matches[0]
        if len(rows) == 1 and rows[0][2] == self.environment["gpu"]:
            return rows[0]
        # Logical CUDA indices can differ from nvidia-smi physical indices.
        visible = os.getenv("CUDA_VISIBLE_DEVICES", "").split(",")
        device = self.environment["device"]
        physical = visible[device].strip() if device < len(visible) and visible[device].strip() else str(device)
        matches = [row for row in rows if row[2] == self.environment["gpu"] and
                   (row[0] == physical or _normalized_uuid(row[1]) == _normalized_uuid(physical))]
        if len(matches) == 1:
            return matches[0]
        matches = [row for row in rows if row[2] == self.environment["gpu"]]
        if len(matches) == 1:
            return matches[0]
        raise RuntimeError("Không xác định được GPU đang chạy trong danh sách nvidia-smi.")

    def _sample(self):
        query = "index,uuid,name,utilization.gpu,utilization.memory,memory.used,power.draw,power.limit,clocks.sm,clocks.mem,temperature.gpu,pstate,driver_version"
        try:
            completed = subprocess.run(
                ["nvidia-smi", f"--query-gpu={query}",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True,
                timeout=2, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if completed.returncode:
                message = (completed.stderr or completed.stdout).strip()[:300]
                raise RuntimeError(f"nvidia-smi lỗi {completed.returncode}: {message}")
            rows = [[part.strip() for part in row]
                    for row in csv.reader(io.StringIO(completed.stdout)) if row]
            if not rows:
                raise RuntimeError("nvidia-smi không trả dữ liệu GPU.")
            if any(len(row) != 13 for row in rows):
                raise RuntimeError("nvidia-smi trả số cột không đúng định dạng.")
            selected = self._select_gpu(rows)
            fields = selected[3:]
            keys = ("gpu_pct", "memory_busy_pct", "vram_mib", "power_w", "power_limit_w",
                    "sm_mhz", "memory_mhz", "temperature_c", "pstate", "driver")
            sample = {"physical_gpu_index": selected[0], "gpu_uuid": selected[1]}
            for key, value in zip(keys, fields):
                if key in {"driver", "pstate"}:
                    sample[key] = value
                    continue
                try:
                    sample[key] = float(value)
                except ValueError:
                    sample[key] = value
            self.telemetry_error = None
        except (OSError, subprocess.SubprocessError, RuntimeError, IndexError) as exc:
            sample = {}
            self.telemetry_error = str(exc)
        if psutil:
            try:
                sample["process_ram_gib"] = psutil.Process().memory_info().rss / 2**30
                sample["system_ram_used_gib"] = psutil.virtual_memory().used / 2**30
            except psutil.Error:
                pass
        sample["elapsed_s"] = time.perf_counter() - self.started
        with self._lock:
            self.samples.append(sample)
            self._snapshot = sample

    def _poll(self):
        while not self._stop.is_set():
            self._sample()
            if self._stop.wait(1):
                break

    def _wrap(self, owner, name, label):
        original = getattr(owner, name)
        had_instance_value = name in owner.__dict__
        self._originals.append((owner, name, original, had_instance_value))

        def timed(*args, **kwargs):
            self.stage = label
            _sync()
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                _sync()
                self.phases[label] = self.phases.get(label, 0) + time.perf_counter() - start

        setattr(owner, name, timed)

    def _before_transformer(self, module, args):
        if self._denoise_start is None:
            _sync()
            self._denoise_start = self._last_step = time.perf_counter()
        self.stage = "Sinh ảnh / denoising"

    def on_step_end(self, pipeline, index, timestep, callback_kwargs):
        _sync()
        now = time.perf_counter()
        if self._last_step is not None:
            self.step_seconds.append(now - self._last_step)
        self._last_step = now
        self.phases["Sinh ảnh / denoising"] = now - self._denoise_start
        self.stage = f"Sinh ảnh: đã xong {len(self.step_seconds)} bước"
        return callback_kwargs

    def __enter__(self):
        global _latest
        self.started = time.perf_counter()
        self.running = True
        torch.cuda.reset_peak_memory_stats()
        self._wrap(self.pipe, "encode_prompt", "Mã hóa prompt / ảnh")
        self._wrap(self.pipe, "prepare_latents", "Chuẩn bị latent / VAE ảnh tham chiếu")
        self._wrap(self.pipe.vae, "decode", "Giải mã VAE")
        self._hook = self.pipe.transformer.register_forward_pre_hook(self._before_transformer)
        self._thread = threading.Thread(target=self._poll, daemon=True)
        with _latest_lock:
            _latest = self
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            _sync()
            self.total_s = time.perf_counter() - self.started
            self.peak_allocated_gib = torch.cuda.max_memory_allocated() / 2**30
            self.peak_reserved_gib = torch.cuda.max_memory_reserved() / 2**30
        finally:
            self._hook.remove()
            for owner, name, original, had_instance_value in reversed(self._originals):
                if had_instance_value:
                    setattr(owner, name, original)
                else:
                    delattr(owner, name)
            self._stop.set()
            self._thread.join(timeout=2.5)
            self.running = False
            self.stage = "Lỗi khi tạo ảnh" if exc_type else "Hoàn tất"

    def report(self):
        with self._lock:
            samples = list(self.samples)
        summary = {}
        for key in ("gpu_pct", "memory_busy_pct", "vram_mib", "power_w", "sm_mhz",
                    "memory_mhz", "temperature_c", "process_ram_gib", "system_ram_used_gib"):
            values = [sample[key] for sample in samples if isinstance(sample.get(key), (int, float))]
            if values:
                summary[key] = {"mean": round(sum(values) / len(values), 2), "max": round(max(values), 2)}
        return {
            "environment": self.environment,
            "phase_seconds": {key: round(value, 3) for key, value in self.phases.items()},
            "pipeline_seconds": round(getattr(self, "total_s", 0), 3),
            "step_seconds": [round(value, 3) for value in self.step_seconds],
            "torch_peak_allocated_gib": round(getattr(self, "peak_allocated_gib", 0), 3),
            "torch_peak_reserved_gib": round(getattr(self, "peak_reserved_gib", 0), 3),
            "gpu_sample_summary": summary, "samples": samples,
            "telemetry_error": self.telemetry_error,
        }

    def text(self):
        env = self.environment
        lines = [f"GPU: {env['gpu']} | VRAM tổng: {env['vram_total_gib']} GiB",
                 f"Torch: {env['torch']} | CUDA: {env['cuda']} | Diffusers: {env['diffusers']}",
                 f"Memory mode: {env['memory_mode']} | Yêu cầu kernel GGUF: {env['gguf_cuda_kernels_requested']}",
                 f"Trạng thái: {self.stage}"]
        with self._lock:
            sample = dict(self._snapshot)
        for label, keys in (
            ("GPU", (("gpu_pct", "% tải"), ("memory_busy_pct", "% hoạt động bộ nhớ"), ("vram_mib", "MiB VRAM"))),
            ("Điện / xung", (("power_w", "W"), ("power_limit_w", "W giới hạn"), ("sm_mhz", "MHz SM"), ("memory_mhz", "MHz bộ nhớ"))),
            ("Nhiệt / RAM", (("temperature_c", "°C"), ("pstate", "P-state"), ("process_ram_gib", "GiB RAM tiến trình"), ("system_ram_used_gib", "GiB RAM toàn máy"))),
        ):
            parts = [f"{sample[key]:.2f} {unit}" if isinstance(sample.get(key), (int, float))
                     else f"{sample[key]} {unit}" for key, unit in keys if key in sample]
            if parts:
                lines.append(f"{label}: " + " | ".join(parts))
        if self.telemetry_error:
            lines.append("Không có telemetry nvidia-smi: " + self.telemetry_error)
        if "driver" in sample:
            lines.append(f"Driver NVIDIA: {sample['driver']}")
        if not self.running:
            lines.append("\nThời gian từng giai đoạn:")
            lines.extend(f"• {label}: {seconds:.2f}s" for label, seconds in self.phases.items())
            other = max(0, getattr(self, "total_s", 0) - sum(self.phases.values()))
            lines.append(f"• Xử lý pipeline khác: {other:.2f}s")
            lines.append(f"• Tổng pipeline: {getattr(self, 'total_s', 0):.2f}s (không tính tải LoRA / lưu PNG)")
            if self.step_seconds:
                lines.append(f"• Trung bình: {sum(self.step_seconds) / len(self.step_seconds):.3f} s/step")
                lines.append(f"• Bước đầu (prefill): {self.step_seconds[0]:.3f}s")
                if len(self.step_seconds) > 1:
                    lines.append(f"• Các bước sau: {sum(self.step_seconds[1:]) / (len(self.step_seconds) - 1):.3f} s/step")
            lines.append(f"VRAM đỉnh PyTorch: {getattr(self, 'peak_allocated_gib', 0):.2f} GiB allocated / {getattr(self, 'peak_reserved_gib', 0):.2f} GiB reserved")
            summary = self.report()["gpu_sample_summary"]
            lines.append("Mẫu GPU trong pipeline (trung bình / đỉnh):")
            for key, label, unit in (("gpu_pct", "GPU tải", "%"), ("power_w", "Công suất", "W"),
                                      ("temperature_c", "Nhiệt độ", "°C"), ("vram_mib", "VRAM", "MiB")):
                if key in summary:
                    lines.append(f"• {label}: {summary[key]['mean']:.1f} / {summary[key]['max']:.1f} {unit}")
        return "\n".join(lines)


def latest_diagnostics():
    with _latest_lock:
        current = _latest
    return current.text() if current else "Chạy một lượt tạo ảnh để xem môi trường, thời gian và telemetry GPU tại đây."
