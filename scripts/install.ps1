# scrt-evolve install script for Windows (PowerShell, x86_64)
# Usage: iwr https://raw.githubusercontent.com/nousresearch/scrt-evolve/main/scripts/install.ps1 | iex
# Or: powershell -ExecutionPolicy Bypass -File install.ps1

param(
    [string]$InstallDir = "$env:USERPROFILE\.local\bin",
    [string]$VenvDir = "$env:LOCALAPPDATA\scrt-evolve\venv",
    [string]$RepoOwner = "nousresearch",
    [string]$RepoName = "scrt-evolve",
    [switch]$Cpu,
    [switch]$Cuda,
    [switch]$WhatIf
)

$ErrorActionPreference = "Stop"

# Colors for output (PowerShell ANSI escape sequences)
function Write-Info { param([string]$Message) Write-Host "[INFO] $Message" -ForegroundColor Green }
function Write-Warn { param([string]$Message) Write-Host "[WARN] $Message" -ForegroundColor Yellow }
function Write-Error { param([string]$Message) Write-Host "[ERROR] $Message" -ForegroundColor Red }

# Detect architecture
function Get-Platform {
    $arch = $env:PROCESSOR_ARCHITECTURE
    if ($arch -ne "AMD64") {
        Write-Error "This script supports x86_64 Windows only. Detected: $arch"
        exit 1
    }
    return "windows-x86_64"
}

# Detect Python
function Get-Python {
    Write-Info "Detecting Python..."
    
    $pythonCmd = $null
    
    # Check common Python locations
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
    if (-not $pythonCmd) {
        $pythonCmd = Get-Command python3 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
    }
    
    if (-not $pythonCmd) {
        # Check in common locations
        $possiblePaths = @(
            "$env:ProgramFiles\Python*\python.exe",
            "$env:LOCALAPPDATA\Programs\Python\Python*\python.exe",
            "$env:USERPROFILE\AppData\Local\Programs\Python\Python*\python.exe"
        )
        
        foreach ($path in $possiblePaths) {
            $found = Get-ChildItem -Path $path -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($found) {
                $pythonCmd = $found.FullName
                break
            }
        }
    }
    
    if (-not $pythonCmd) {
        Write-Error "Python 3.9+ not found. Please install Python from https://python.org"
        exit 1
    }
    
    # Check version
    $versionOutput = & $pythonCmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
    if (-not $versionOutput) {
        Write-Error "Python found but could not determine version"
        exit 1
    }
    
    $major = [int]($versionOutput.Split('.')[0])
    $minor = [int]($versionOutput.Split('.')[1])
    
    if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 9)) {
        Write-Error "Python 3.9+ required. Detected: $versionOutput"
        exit 1
    }
    
    Write-Info "Found Python $versionOutput at $pythonCmd"
    return $pythonCmd
}

# Detect CUDA
function Get-Accelerator {
    param([string]$PythonCmd)
    
    # Check for CUDA via nvidia-smi first
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($nvidiaSmi) {
        try {
            $null = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Info "CUDA detected via nvidia-smi"
                return "cuda"
            }
        } catch {
            # nvidia-smi exists but failed
        }
    }
    
    Write-Info "No CUDA detected, using CPU"
    return "cpu"
}

# Ensure installation directory
function Initialize-InstallDir {
    param([string]$Dir)
    
    Write-Info "Ensuring installation directory exists: $Dir"
    
    if (-not (Test-Path $Dir)) {
        New-Item -ItemType Directory -Path $Dir -Force | Out-Null
    }
    
    # Check if directory is on PATH
    $pathEntry = $env:PATH -split ';' | Where-Object { $_ -eq $Dir }
    if (-not $pathEntry) {
        Write-Warn "$Dir is not in your PATH"
        Write-Host ""
        Write-Host "To add to your PATH, add this to your PowerShell profile (`$PROFILE):"
        Write-Host "  `$env:PATH = `"$Dir;`$env:PATH`""
        Write-Host ""
    }
}

# Fetch latest release
function Get-LatestRelease {
    param([string]$Owner, [string]$Name)
    
    Write-Info "Fetching latest release from GitHub..."
    
    $apiUrl = "https://api.github.com/repos/$Owner/$Name/releases/latest"
    
    try {
        $response = Invoke-RestMethod -Uri $apiUrl -UseBasicParsing
        $tag = $response.tag_name
    } catch {
        Write-Warn "Could not fetch latest release. Using fallback."
        return "v0.1.0"
    }
    
    Write-Info "Latest release: $tag"
    return $tag
}

# Download binary
function Install-Binary {
    param([string]$Tag, [string]$Platform, [string]$OutDir, [string]$Owner, [string]$Name)
    
    Write-Info "Downloading scrt-evolve $Tag for $Platform..."
    
    $downloadUrl = "https://github.com/$Owner/$Name/releases/download/$Tag/evolve-$Platform.exe"
    $outputPath = Join-Path $OutDir "evolve.exe"
    
    if ($WhatIf) {
        Write-Info "[WhatIf] Would download from $downloadUrl to $outputPath"
        return
    }
    
    try {
        Invoke-WebRequest -Uri $downloadUrl -OutFile $outputPath -UseBasicParsing
        Write-Info "Downloaded to $outputPath"
    } catch {
        Write-Error "Download failed: $_"
        Write-Error "The release may not have artifacts for this platform."
        Write-Error "You may need to build from source: cargo build --release -p scrt-evolve"
        exit 1
    }
}

# Create virtual environment
function Initialize-Venv {
    param([string]$PythonCmd, [string]$VenvPath, [string]$Accelerator)
    
    Write-Info "Setting up Python virtual environment at $VenvPath"
    
    if (Test-Path $VenvPath) {
        Write-Info "Virtual environment already exists, skipping creation"
        return
    }
    
    $venvParent = Split-Path $VenvPath -Parent
    if (-not (Test-Path $venvParent)) {
        New-Item -ItemType Directory -Path $venvParent -Force | Out-Null
    }
    
    Write-Info "Creating virtual environment..."
    & $PythonCmd -m venv $VenvPath
    
    $pipCmd = Join-Path $VenvPath "Scripts\pip.exe"
    
    Write-Info "Installing scrt-evolve-ml[$Accelerator]..."
    & $pipCmd install --upgrade pip
    
    & $pipCmd install "scrt-evolve-ml[$Accelerator]"
    
    Write-Info "Installation complete!"
}

# Main installation flow
function Main {
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host "  scrt-evolve Installer (Windows x86_64)" -ForegroundColor Cyan
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host ""
    
    # Detect platform
    $platform = Get-Platform
    
    # Ensure install directory
    Initialize-InstallDir -Dir $InstallDir
    
    # Fetch latest release
    $tag = Get-LatestRelease -Owner $RepoOwner -Name $RepoName
    
    # Download binary
    Install-Binary -Tag $tag -Platform $platform -OutDir $InstallDir -Owner $RepoOwner -Name $RepoName
    
    # Detect or use specified Python
    $pythonCmd = Get-Python
    
    # Detect or use specified accelerator
    if ($Cpu -and -not $Cuda) {
        $accelerator = "cpu"
    } elseif ($Cuda) {
        $accelerator = "cuda"
    } else {
        $accelerator = Get-Accelerator -PythonCmd $pythonCmd
    }
    
    # Create virtual environment
    Initialize-Venv -PythonCmd $pythonCmd -VenvPath $VenvDir -Accelerator $accelerator
    
    $venvPython = Join-Path $VenvDir "Scripts\python.exe"
    
    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Green
    Write-Host "  Installation complete!" -ForegroundColor Green
    Write-Host "==========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:"
    Write-Host "  1. Add $InstallDir to your PATH if not already"
    Write-Host "  2. Set environment variable:"
    Write-Host "     `$env:SCRT_EVOLVE_PYTHON = `"$venvPython`""
    Write-Host "  3. Run: evolve doctor"
    Write-Host ""
    Write-Host "Note: For LoRA training with SSM/Mamba, use WSL2 + CUDA."
    Write-Host "See PORTABILITY.md for details."
}

# Run main
Main