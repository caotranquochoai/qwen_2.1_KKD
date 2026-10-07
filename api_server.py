"""Standalone OpenAI-compatible Images API; leaves the original Gradio app intact."""

import asyncio
import base64
from collections import OrderedDict
from contextlib import asynccontextmanager
import hmac
import importlib
import io
import logging
import os
from pathlib import Path
import time
from typing import Literal
import uuid
import warnings

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException
from workflow_support import MAX_GARMENT_IMAGES, MULTI_OUTFIT_MODE

MODEL_NAME = "qwen-image-2.1"
MAX_UPLOAD = 10 * 1024 * 1024
FILE_TTL = 3600
MAX_FILES = 64
logger = logging.getLogger(__name__)


class ImageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=4000)
    model: str = MODEL_NAME
    n: Literal[1] = 1
    size: str = "1024x1024"
    response_format: Literal["url", "b64_json"] = "b64_json"
    background: Literal["auto", "opaque", "transparent"] = "auto"
    user: str | None = None
    steps: int = Field(default=30, ge=4, le=40)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)
    lora_adapter: str | None = None
    lora_strength: float = Field(default=1.0, ge=0, le=1.5)
    custom_repo: str = ""
    custom_file: str = ""
    edit_mode: Literal["auto", "outfit"] = "auto"

    @field_validator("n", mode="before")
    @classmethod
    def multipart_n(cls, value):
        # Multipart fields arrive as strings; the SDK sends n="1".
        return 1 if value == "1" else value

    @field_validator("prompt")
    @classmethod
    def nonempty_prompt(cls, value):
        if not value.strip():
            raise ValueError("Prompt must not be blank.")
        return value


def create_api(backend=None, *, api_key=None):
    """backend injection allows integration without loading a second model."""
    key = os.environ.get("QWEN_API_KEY", "") if api_key is None else api_key

    @asynccontextmanager
    async def lifespan(server):
        # app.py loads weights once at import; its __main__ launch is not executed.
        server.state.backend = backend or await run_in_threadpool(importlib.import_module, "app")
        server.state.inference_lock = asyncio.Lock()
        server.state.files = OrderedDict()
        yield
        server.state.files.clear()

    server = FastAPI(title="Qwen Image OpenAI-compatible API", lifespan=lifespan)

    def authorize(request: Request):
        if key and not hmac.compare_digest(
            request.headers.get("authorization", "").encode("utf-8"),
            f"Bearer {key}".encode("utf-8"),
        ):
            raise HTTPException(401, "Invalid API key.")

    @server.exception_handler(StarletteHTTPException)
    async def http_error(request, exc):
        return JSONResponse(status_code=exc.status_code, content={"error": {
            "message": str(exc.detail),
            "type": "authentication_error" if exc.status_code == 401 else "invalid_request_error",
            "param": None, "code": None,
        }})

    @server.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        error = exc.errors()[0]
        return JSONResponse(status_code=400, content={"error": {
            "message": error["msg"], "type": "invalid_request_error",
            "param": str(error["loc"][-1]), "code": None,
        }})

    def prune_files():
        files = server.state.files
        now = time.time()
        for token, (expires, _) in list(files.items()):
            if expires <= now:
                del files[token]
        while len(files) > MAX_FILES:
            files.popitem(last=False)

    def invoke(options, images, aspect, adapter):
        source = server.state.backend
        is_outfit = len(images) > 2 or options.edit_mode == "outfit"
        mode = (MULTI_OUTFIT_MODE if is_outfit else
                "Transform & Swap (2 Refs)" if len(images) == 2 else
                "Edit Image (1 Ref)" if images else
                "Transparent PNG" if options.background == "transparent" else "Text to Image")
        # Hold the shared backend lock across loading and generation so UI model
        # switches cannot unload the pipeline between these operations.
        with source._MODEL_LOCK:
            source.ensure_model_loaded()
            return generate_output(source, options, images, aspect, adapter, mode, is_outfit)

    def generate_output(source, options, images, aspect, adapter, mode, is_outfit):
        path, _, seed, details = source.generate(
            prompt=options.prompt, mode=mode,
            ref_image_1=images[0] if images else None,
            ref_image_2=images[1] if len(images) == 2 else None,
            lora_adapter=adapter, lora_strength=options.lora_strength,
            custom_repo=options.custom_repo, custom_file=options.custom_file,
            aspect_ratio=aspect, steps=options.steps,
            seed=options.seed if options.seed is not None else 42,
            randomize_seed=options.seed is None,
            garment_images=images[1:] if is_outfit else None,
        )
        output = Path(path)
        try:
            return output.read_bytes(), seed, details
        finally:
            # Only remove the output created by generate(), never a directory tree.
            output.unlink(missing_ok=True)
            try:
                output.parent.rmdir()
            except OSError:
                pass

    async def infer(request, options, images):
        source = server.state.backend
        if options.edit_mode == "outfit" and len(images) < 2:
            raise HTTPException(400, "Outfit editing requires a person image and at least one garment image.")
        if options.model not in {MODEL_NAME, source.MODEL_ID}:
            raise HTTPException(400, f"Unknown model. Use {MODEL_NAME}.")
        sizes = {f"{w}x{h}": name for name, (w, h) in source.SIZES.items()}
        size = "1024x1024" if options.size == "auto" else options.size
        if size not in sizes:
            raise HTTPException(400, f"Supported sizes: {', '.join(sizes)}.")
        adapter = options.lora_adapter or source.NONE_LORA
        if adapter not in source.LORA_CHOICES:
            raise HTTPException(400, "Unknown lora_adapter; see /v1/qwen/config.")
        if adapter == "Custom HuggingFace LoRA..." and not (
            options.custom_repo.strip() and options.custom_file.strip()
        ):
            raise HTTPException(400, "Custom LoRA requires custom_repo and custom_file.")
        if images and options.background == "transparent":
            raise HTTPException(400, "Transparent background is supported only for generation.")
        async with server.state.inference_lock:
            try:
                png, seed, details = await run_in_threadpool(
                    invoke, options, images, sizes[size], adapter
                )
            except Exception:
                logger.exception("Image inference failed")
                return JSONResponse(status_code=500, content={"error": {
                    "message": "Image inference failed. See the server log.",
                    "type": "server_error", "param": None, "code": None,
                }})
        if options.response_format == "b64_json":
            item = {"b64_json": base64.b64encode(png).decode("ascii")}
        else:
            token = uuid.uuid4().hex
            server.state.files[token] = (time.time() + FILE_TTL, png)
            prune_files()
            base = os.environ.get("QWEN_API_PUBLIC_URL", "").rstrip("/")
            url = f"{base}/v1/files/{token}" if base else str(request.url_for("image_file", token=token))
            item = {"url": url}
        return {"created": int(time.time()), "data": [item],
                "qwen": {"seed": seed, "details": details}}

    @server.get("/v1/models", dependencies=[Depends(authorize)])
    async def models():
        return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model",
                 "created": 0, "owned_by": "local"}]}

    @server.get("/v1/qwen/config", dependencies=[Depends(authorize)])
    async def config():
        source = server.state.backend
        return {"model": MODEL_NAME, "sizes": [f"{w}x{h}" for w, h in source.SIZES.values()],
                "loras": source.LORA_CHOICES, "steps": {"min": 4, "max": 40}, "n": 1,
                "max_reference_images": MAX_GARMENT_IMAGES + 1,
                "model_selection": source.MODEL_SELECTION,
                "model_loaded": source.pipe is not None, "model_status": source.MODEL_STATUS,
                "edit_modes": ["auto", "outfit"]}

    @server.post("/v1/images/generations", dependencies=[Depends(authorize)])
    async def generations(options: ImageRequest, request: Request):
        return await infer(request, options, [])

    @server.post("/v1/images/edits", dependencies=[Depends(authorize)])
    async def edits(request: Request):
        if not request.headers.get("content-type", "").startswith("multipart/form-data"):
            raise HTTPException(400, "Edits require multipart/form-data with image or image[].")
        images = []
        async with request.form(max_files=MAX_GARMENT_IMAGES + 1, max_fields=24, max_part_size=MAX_UPLOAD) as form:
            fields = {}
            uploads = []
            for name, value in form.multi_items():
                if isinstance(value, UploadFile):
                    if name not in {"image", "image[]"}:
                        raise HTTPException(400, f"Unsupported file field: {name}. Masks are not supported.")
                    uploads.append(value)
                else:
                    if name in fields:
                        raise HTTPException(400, f"Duplicate field: {name}.")
                    fields[name] = value
            if not 1 <= len(uploads) <= MAX_GARMENT_IMAGES + 1:
                raise HTTPException(400, "Provide 1–10 reference images in order.")
            try:
                options = ImageRequest.model_validate(fields)
            except ValidationError as exc:
                error = exc.errors()[0]
                raise HTTPException(400, f"{error['loc'][0]}: {error['msg']}") from exc
            for upload in uploads:
                raw = await upload.read(MAX_UPLOAD + 1)
                if len(raw) > MAX_UPLOAD:
                    raise HTTPException(413, "Each image must be at most 10 MiB.")
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", Image.DecompressionBombWarning)
                        with Image.open(io.BytesIO(raw)) as image:
                            if image.width * image.height > 16_000_000:
                                raise HTTPException(400, "Reference image must be at most 16 megapixels.")
                            images.append(ImageOps.exif_transpose(image).convert("RGBA"))
                except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
                        Image.DecompressionBombWarning) as exc:
                    raise HTTPException(400, "Invalid or oversized reference image.") from exc
        try:
            return await infer(request, options, images)
        finally:
            for image in images:
                image.close()

    @server.get("/v1/files/{token}", name="image_file", dependencies=[Depends(authorize)])
    async def image_file(token: str):
        prune_files()
        entry = server.state.files.get(token)
        if entry is None:
            raise HTTPException(404, "Image expired or not found.")
        return Response(entry[1], media_type="image/png")

    return server


api = create_api()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(api, host=os.environ.get("QWEN_API_HOST", "127.0.0.1"),
                port=int(os.environ.get("QWEN_API_PORT", "8000")))
