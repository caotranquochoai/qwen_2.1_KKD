# Windows 10/11 x64, PowerShell 5.1+. Run from a downloaded/cloned project.
# CUDA wheels: https://download.pytorch.org/whl/cu132/torch/
[CmdletBinding()]
param(
    [string]$PythonPath = '',
    [ValidateSet('None', 'UI', 'API')]
    [string]$Launch = 'None',
    [switch]$SkipPrerequisiteInstall
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    if ($Arguments.Count -ge 2 -and $Arguments[0] -eq '-c') {
        # Send Python snippets through stdin to preserve quotes on PowerShell 5.1.
        $pythonSnippet = $Arguments[1]
        $remainingArguments = @($Arguments | Select-Object -Skip 2)
        $pythonSnippet | & $Executable '-' @remainingArguments
    } else {
        & $Executable @Arguments
    }
    if ($LASTEXITCODE -ne 0) {
        throw "$Executable failed (exit code $LASTEXITCODE). Installation stopped."
    }
}

function Update-ProcessPath {
    $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' +
        [Environment]::GetEnvironmentVariable('Path', 'User')
}

function Install-Prerequisite {
    param([string]$Id)
    if ($SkipPrerequisiteInstall) {
        throw "Missing $Id. Install it manually, then rerun this script."
    }
    $wingetCommand = Get-Command winget.exe -ErrorAction SilentlyContinue
    if (-not $wingetCommand) {
        throw "WinGet is missing. Install Python 3.12 x64 and Git for Windows, then rerun."
    }
    Write-Host "Installing $Id with WinGet..." -ForegroundColor Cyan
    Invoke-Checked $wingetCommand.Source @(
        'install', '--id', $Id, '--exact', '--source', 'winget',
        '--accept-package-agreements', '--accept-source-agreements', '--disable-interactivity'
    )
    Update-ProcessPath
}

function Find-Python312 {
    # Missing launcher versions can write stderr on Windows PowerShell 5.1.
    $ErrorActionPreference = 'Continue'
    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($launcher) {
        $result = & $launcher.Source -3.12 -c 'import sys; print(sys.executable)' 2>$null
        if ($LASTEXITCODE -eq 0 -and $result) {
            return ([string]($result | Select-Object -Last 1)).Trim()
        }
    }
    $candidates = @(
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'),
        (Join-Path $env:ProgramFiles 'Python312\python.exe')
    )
    $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($pythonCommand -and $pythonCommand.Source -notlike '*\WindowsApps\*') {
        $candidates += $pythonCommand.Source
    }
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            $version = & $candidate -c 'import sys; print(sys.version_info.major, sys.version_info.minor, sep=chr(46))' 2>$null
            if ($LASTEXITCODE -eq 0 -and $version -eq '3.12') { return $candidate }
        }
    }
    return $null
}

try {
    if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT -or
        -not [Environment]::Is64BitOperatingSystem) {
        throw 'This installer requires 64-bit Windows.'
    }
    foreach ($requiredFile in @('app.py', 'api_server.py', 'requirements.txt', 'requirements-api.txt')) {
        if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot $requiredFile) -PathType Leaf)) {
            throw "Missing $requiredFile. Download or clone the complete repository first."
        }
    }
    if (-not (Get-Command git.exe -ErrorAction SilentlyContinue)) {
        Install-Prerequisite 'Git.Git'
    }
    if (-not (Get-Command git.exe -ErrorAction SilentlyContinue)) {
        throw 'Git is not on PATH. Reopen PowerShell and rerun the installer.'
    }

    $venvDirectory = Join-Path $PSScriptRoot 'env'
    $venvPython = Join-Path $venvDirectory 'Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
        if (Test-Path -LiteralPath $venvDirectory) {
            throw 'env exists but has no working Python. Rename that directory and rerun; no files were deleted.'
        }
        if ($PythonPath) {
            $basePython = (Get-Command $PythonPath -ErrorAction Stop).Source
        } else {
            $basePython = Find-Python312
            if (-not $basePython) {
                Install-Prerequisite 'Python.Python.3.12'
                $basePython = Find-Python312
            }
        }
        if (-not $basePython) {
            throw 'Python 3.12 was not found. Reopen PowerShell or use -PythonPath with its full path.'
        }
        Invoke-Checked $basePython @('-c', 'import sys, struct; assert sys.version_info[:2] == (3, 12) and struct.calcsize("P") == 8, "Python 3.12 x64 is required"')
        Write-Host 'Creating env...' -ForegroundColor Cyan
        Invoke-Checked $basePython @('-m', 'venv', $venvDirectory)
    }
    Invoke-Checked $venvPython @('-c', 'import sys, struct; assert sys.version_info[:2] == (3, 12) and struct.calcsize("P") == 8, "Existing env must use Python 3.12 x64; rename it to preserve it and rerun"')
    Invoke-Checked $venvPython @('-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel')

    Write-Host 'Installing PyTorch 2.14.0 with CUDA 13.2 (large download)...' -ForegroundColor Cyan
    Invoke-Checked $venvPython @(
        '-m', 'pip', 'install', '--upgrade', 'torch==2.14.0+cu132', 'torchvision',
        '--index-url', 'https://download.pytorch.org/whl/cu132'
    )
    # Keep dependency resolution from replacing CUDA torch/torchvision with CPU wheels.
    $constraintFile = [IO.Path]::GetTempFileName()
    try {
        Invoke-Checked $venvPython @(
            '-c', 'import importlib.metadata as m, pathlib, sys; pathlib.Path(sys.argv[1]).write_text("\n".join(n + "==" + m.version(n) for n in ("torch", "torchvision")) + "\n", encoding="utf-8")',
            $constraintFile
        )
        Write-Host 'Installing project, UI and API dependencies...' -ForegroundColor Cyan
        Invoke-Checked $venvPython @(
            '-m', 'pip', 'install', '-c', $constraintFile,
            '-r', (Join-Path $PSScriptRoot 'requirements.txt'),
            '-r', (Join-Path $PSScriptRoot 'requirements-api.txt')
        )
    } finally {
        Remove-Item -LiteralPath $constraintFile -ErrorAction SilentlyContinue
    }
    # Installation check only; does not load model weights or generate an image.
    Invoke-Checked $venvPython @('-c', 'import torch; print("Torch:", torch.__version__, "CUDA:", torch.version.cuda); assert torch.cuda.is_available(), "CUDA is unavailable. Install/update the NVIDIA driver and rerun."; print("GPU:", torch.cuda.get_device_name(0)); print("VRAM: %.2f GiB" % (torch.cuda.get_device_properties(0).total_memory / 2**30))')

    Write-Host 'Installation completed. Model downloads begin when you click Load model.' -ForegroundColor Green
    Write-Host ('UI:  & "{0}" "{1}"' -f $venvPython, (Join-Path $PSScriptRoot 'app.py'))
    Write-Host ('API: & "{0}" "{1}"' -f $venvPython, (Join-Path $PSScriptRoot 'api_server.py'))
    if ($Launch -ne 'None') {
        $entryFile = if ($Launch -eq 'UI') { 'app.py' } else { 'api_server.py' }
        Push-Location -LiteralPath $PSScriptRoot
        try {
            Invoke-Checked $venvPython @((Join-Path $PSScriptRoot $entryFile))
        } finally {
            Pop-Location
        }
    }
} catch {
    Write-Error ("Setup failed: " + $_.Exception.Message) -ErrorAction Continue
    exit 1
}
