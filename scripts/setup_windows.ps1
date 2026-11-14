# Laptop setup for the generative-models lecture, on Windows.
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
#
# What it does, all inside this project folder:
#   1. finds Python 3.10 or newer through the "py" launcher (no version is hard-coded; the newest is used)
#   2. creates .venv here, or reuses it if it is already there and new enough
#   3. upgrades pip inside .venv
#   4. installs requirements.txt and the genroofs package (editable) into .venv
#   5. runs the self-check: python -m genroofs.check
#
# It needs no administrator rights and installs nothing globally. "-ExecutionPolicy Bypass" applies to this
# one run only; no system setting changes. (pip keeps its usual download cache in your user cache folder.)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$Project = (Get-Location).Path
$Venv = Join-Path $Project ".venv"
$VenvPython = Join-Path $Venv "Scripts\python.exe"

function Say([string]$Text) {
    Write-Host ""
    Write-Host "==> $Text"
}

function Fail([string]$Text) {
    Write-Host ""
    Write-Host "STOPPED: $Text" -ForegroundColor Red
    exit 1
}

function Tidy([string]$Path) {
    # Show paths with ~ instead of the home folder, so no user name is printed.
    $HomeDir = $env:USERPROFILE
    if ($HomeDir -and $Path.StartsWith($HomeDir, [System.StringComparison]::OrdinalIgnoreCase)) {
        return "~" + $Path.Substring($HomeDir.Length)
    }
    return $Path
}

function Test-NewEnough([string]$Exe, [string[]]$Extra) {
    # True when this Python is 3.10 or newer. A missing or broken Python simply counts as "no".
    try {
        $null = & $Exe @Extra -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" 2>$null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

function Invoke-Venv([string]$Problem, [string[]]$Arguments) {
    & $VenvPython @Arguments
    if ($LASTEXITCODE -ne 0) { Fail $Problem }
}

Say "Project folder: $(Tidy $Project)"

if ((Test-Path $VenvPython) -and (Test-NewEnough $VenvPython @())) {
    Say "Reusing $(Tidy $Venv) ($(& $VenvPython --version))"
} else {
    if (Test-Path $Venv) {
        Fail "$(Tidy $Venv) exists but has no Python 3.10 or newer. Rename or delete that folder, then run this again."
    }
    $PyExe = $null
    $PyArgs = @()
    if ((Get-Command py -ErrorAction SilentlyContinue) -and (Test-NewEnough "py" @("-3"))) {
        $PyExe = "py"
        $PyArgs = @("-3")
    } elseif ((Get-Command python -ErrorAction SilentlyContinue) -and (Test-NewEnough "python" @())) {
        $PyExe = "python"
    }
    if (-not $PyExe) {
        Fail "Python 3.10 or newer was not found. Install it from https://www.python.org/downloads/windows/ (see docs\SETUP.md), then open a new PowerShell window and run this again."
    }
    Say "Creating $(Tidy $Venv) with $(& $PyExe @PyArgs --version)"
    & $PyExe @PyArgs -m venv $Venv
    if ($LASTEXITCODE -ne 0) { Fail "Could not create the virtual environment." }
}

Say "Upgrading pip inside .venv"
Invoke-Venv "pip could not upgrade itself. Check the internet connection." @("-m", "pip", "install", "--upgrade", "pip")

Say "Installing the lab's packages into .venv (torch is large; the first time can take several minutes)"
Invoke-Venv "Installing requirements.txt failed. Scroll up for pip's message." @("-m", "pip", "install", "-r", "requirements.txt")

Say "Installing genroofs from this folder (editable)"
Invoke-Venv "Installing genroofs failed. Scroll up for pip's message." @("-m", "pip", "install", "-e", ".", "--no-deps")

Say "Running the self-check"
& $VenvPython -m genroofs.check
if ($LASTEXITCODE -ne 0) {
    Fail "The self-check listed something to fix (see above). Fix it, then run this script again."
}

Write-Host ""
Write-Host "Done. To use this setup later, open PowerShell in this folder and run, for example:"
Write-Host "    .venv\Scripts\python.exe -m pytest -q tests"
Write-Host "    .venv\Scripts\python.exe -m jupyter lab"
