#!/usr/bin/env pwsh
<#
.SYNOPSIS
    One-click deploy script for Orange Pi 5 Pro (run on Windows).

.DESCRIPTION
    Packs deploy_opi5/ (excludes __pycache__/venv/.git) -> scp upload -> ssh extract,
    optionally runs install.sh remotely to install Python deps.

    First time: run with -SetupKey to enable passwordless SSH login.

.EXAMPLE
    .\deploy_to_opi5.ps1 -IP 192.168.1.100
    .\deploy_to_opi5.ps1 -IP 192.168.1.100 -User orangepi -Install
    .\deploy_to_opi5.ps1 -IP 192.168.1.100 -SetupKey
#>
param(
    [Parameter(Mandatory = $true, HelpMessage = "Orange Pi 5 Pro IP address")]
    [string]$IP,

    [string]$User = "opi",          # Armbian: opi/root; official image: orangepi
    [int]$Port = 22,
    [string]$RemoteDir = "~/fall_detection",

    [switch]$Install,               # run install.sh remotely after upload
    [switch]$SetupKey,              # first-run: configure passwordless SSH
    [switch]$SkipUpload             # skip pack/upload, only run remote steps
)

$ErrorActionPreference = "Stop"
$LocalDir = Split-Path -Parent $MyInvocation.MyCommand.Path   # this folder = deploy_opi5

function Step([string]$msg) { Write-Host "`n==> $msg" -ForegroundColor Yellow }

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "  Orange Pi 5 Pro One-Click Deploy" -ForegroundColor Cyan
Write-Host "  Local : $LocalDir" -ForegroundColor DarkGray
Write-Host "  Target: ${User}@${IP}:${Port}  ->  ${RemoteDir}" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

# ── 0. Passwordless SSH setup ──
if ($SetupKey) {
    Step "Configure passwordless SSH login"
    $key = "$env:USERPROFILE\.ssh\id_ed25519_opi5"
    if (-not (Test-Path $key)) {
        ssh-keygen -t ed25519 -f $key -N '' -C "opi5-deploy"
    }
    $pub = (Get-Content "$key.pub" -Raw).Trim()
    Write-Host "  You will be asked for the board password ONCE, then it's passwordless."
    ssh -p $Port "${User}@${IP}" "mkdir -p ~/.ssh && chmod 700 ~/.ssh && echo '$pub' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
    if ($LASTEXITCODE -ne 0) { throw "SSH key setup failed" }
    Write-Host "  OK - passwordless login configured" -ForegroundColor Green
    exit 0
}

if (-not $SkipUpload) {
    # ── 1. Pack ──
    Step "Packing deploy_opi5 (excluding __pycache__/venv/.git)"
    $tarName = "deploy_opi5.tar.gz"
    $tarPath = Join-Path $env:TEMP $tarName
    Push-Location $LocalDir
    try {
        tar -czf $tarPath --exclude="__pycache__" --exclude="venv" --exclude=".git" .
        if ($LASTEXITCODE -ne 0) { throw "tar pack failed" }
    } finally { Pop-Location }
    $sizeMB = [math]::Round((Get-Item $tarPath).Length / 1MB, 2)
    Write-Host "  OK - $tarPath ($sizeMB MB)" -ForegroundColor Green

    # ── 2. Upload ──
    Step "Uploading to board (~/$tarName)"
    scp -P $Port $tarPath "${User}@${IP}:~/$tarName"
    if ($LASTEXITCODE -ne 0) { throw "scp upload failed" }
    Write-Host "  OK - uploaded" -ForegroundColor Green

    # ── 3. Extract remotely ──
    Step "Extracting to $RemoteDir"
    ssh -p $Port "${User}@${IP}" "mkdir -p $RemoteDir && tar -xzf ~/$tarName -C $RemoteDir && rm -f ~/$tarName && echo '--- files ---' && ls $RemoteDir"
    if ($LASTEXITCODE -ne 0) { throw "remote extract failed" }
    Write-Host "  OK - deploy files in place" -ForegroundColor Green
}

# ── 4. Optional: install deps ──
if ($Install) {
    Step "Running install.sh remotely (5-15 min, keep network alive)"
    ssh -p $Port "${User}@${IP}" "cd $RemoteDir && bash install.sh"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  WARN - install.sh returned non-zero (likely MediaPipe failed to install; see SETUP_GUIDE.md)" -ForegroundColor Red
    } else {
        Write-Host "  OK - dependencies installed" -ForegroundColor Green
    }
}

Write-Host "`n==============================================" -ForegroundColor Cyan
Write-Host "  Deploy done! Next steps:" -ForegroundColor Cyan
Write-Host "    ssh ${User}@${IP}" -ForegroundColor White
Write-Host "    cd $RemoteDir && source venv/bin/activate" -ForegroundColor White
Write-Host "    python3 fall_inference.py --benchmark" -ForegroundColor White
Write-Host "==============================================" -ForegroundColor Cyan
