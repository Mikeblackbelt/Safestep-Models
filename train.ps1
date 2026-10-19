# ==============================================================================
# SafeStep Model Training Terminal Helper (Windows PowerShell)
# ==============================================================================
# Native PowerShell equivalent of train.sh for SafeStep models
# (CART, JAW, PHISHING, or ALL). Auto-detects virtual environment and supports
# both interactive and direct CLI workflows.
# Works in Windows PowerShell 5.1 and PowerShell 7+.
#
#   powershell -ExecutionPolicy Bypass -File train.ps1 [COMMAND | MODEL] [OPTIONS...]
# ==============================================================================

$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$CmdArgs = @($args)
$Models = @('cart', 'jaw', 'phishing', 'all')

# Determine Python executable (prioritize virtualenv if present)
$PyBin = $null
$PyPre = @()
$candidates = @()
if ($env:VIRTUAL_ENV) { $candidates += (Join-Path $env:VIRTUAL_ENV 'Scripts\python.exe') }
$candidates += (Join-Path $ScriptDir '.venv\Scripts\python.exe')
$candidates += (Join-Path $ScriptDir 'venv\Scripts\python.exe')
foreach ($c in $candidates) {
    if (Test-Path -LiteralPath $c -PathType Leaf) { $PyBin = $c; break }
}
if (-not $PyBin) {
    if (Get-Command py -ErrorAction SilentlyContinue) { $PyBin = 'py'; $PyPre = @('-3') }
    elseif (Get-Command python -ErrorAction SilentlyContinue) { $PyBin = 'python' }
    else {
        Write-Host 'Error: Python not found (checked $env:VIRTUAL_ENV, .venv, venv, py, python).' -ForegroundColor Red
        exit 1
    }
}

function Invoke-Py {
    param([string[]]$PyArgs)
    $all = @($PyPre) + @($PyArgs)
    & $PyBin @all
    exit $LASTEXITCODE
}

function Write-Row([string]$Cmd, [string]$Desc, [string]$Color) {
    if ($Cmd.Length -lt 32) {
        Write-Host ('  ' + $Cmd.PadRight(32)) -ForegroundColor $Color -NoNewline
        Write-Host $Desc
    } else {
        Write-Host ('  ' + $Cmd) -ForegroundColor $Color
        Write-Host ((' ' * 34) + $Desc)
    }
}

function Show-Help {
    Write-Host 'SafeStep Easy Terminal Trainer' -ForegroundColor White
    Write-Host 'Usage: ' -NoNewline
    Write-Host '.\train.ps1 [COMMAND | MODEL] [OPTIONS...]' -ForegroundColor Cyan
    Write-Host ''
    Write-Host 'Quick Model Commands:' -ForegroundColor White
    Write-Row '.\train.ps1 cart' 'Train CART (AI-text detector)' Green
    Write-Row '.\train.ps1 jaw' 'Train JAW (AI-image detector)' Green
    Write-Row '.\train.ps1 phishing' 'Train PHISHING (phishing detector)' Green
    Write-Row '.\train.ps1 all' 'Train all models concurrently' Green
    Write-Row '.\train.ps1 demo [cart|jaw|phishing|all]' 'Run instant training on demo data' Green
    Write-Host ''
    Write-Host 'Utility & Monitoring Commands:' -ForegroundColor White
    Write-Row '.\train.ps1 prepare [model] [data-dir]' 'Generate sample datasets (model: cart|jaw|phishing|all)' Cyan
    Write-Row '.\train.ps1 download [sources...]' 'Download real phishing datasets (default: hf_emails hf_urls; all = every source)' Cyan
    Write-Row '.\train.ps1 inspect [dir]' 'Inspect a folder and report image/text counts' Cyan
    Write-Row '.\train.ps1 cifake <dir>' 'Import downloaded CIFAKE folder into data/jaw' Cyan
    Write-Row '.\train.ps1 dashboard' 'Train with full-screen live dashboard & metrics' Cyan
    Write-Row '.\train.ps1 estimate [cart|jaw|phishing|all]' 'Preview training time & hardware plan without training' Cyan
    Write-Row '.\train.ps1 history' 'Show history, charts, and metrics of past runs' Cyan
    Write-Row '.\train.ps1 interactive' 'Interactive step-by-step menu wizard' Cyan
    Write-Host ''
    Write-Host 'Common Options:' -ForegroundColor White
    Write-Host '  --data-dir PATH           Custom dataset path (default: data/)'
    Write-Host '  --device [cpu|cuda|mps]   Force compute device (default: auto-detected)'
    Write-Host '  --workers N               Data loader worker threads'
    Write-Host '  --batch-size N            Override batch size'
    Write-Host '  --lr RATE                 Override learning rate'
    Write-Host '  --help                    Show full training CLI help'
}

function Read-Model([string]$Prompt) {
    $m = Read-Host $Prompt
    if (-not $m) { $m = 'all' }
    return $m.ToLower()
}

function Start-Wizard {
    Write-Host '=== SafeStep Interactive Training Wizard ===' -ForegroundColor Cyan
    Write-Host ''
    Write-Host 'Select what you would like to do:'
    Write-Host '  1) Train CART (AI-generated text detector)'
    Write-Host '  2) Train JAW (AI-generated image detector)'
    Write-Host '  3) Train Both (CART & JAW) in parallel'
    Write-Host '  4) Quick Demo training (test pipeline on synthetic data)'
    Write-Host '  5) Prepare sample training datasets in data/'
    Write-Host '  6) Estimate training time only'
    Write-Host '  7) View past training history and metrics'
    Write-Host '  8) Train PHISHING (phishing email/URL detector)'
    Write-Host '  q) Quit'
    Write-Host ''
    $choice = Read-Host 'Enter choice [1-8/q]'
    switch ($choice) {
        '1' { Write-Host "`nStarting CART training..." -ForegroundColor Green; Invoke-Py @('train.py', '--model', 'cart') }
        '2' { Write-Host "`nStarting JAW training..." -ForegroundColor Green; Invoke-Py @('train.py', '--model', 'jaw') }
        '3' { Write-Host "`nStarting parallel training (both models)..." -ForegroundColor Green; Invoke-Py @('train.py', '--model', 'all', '--dashboard') }
        '4' {
            $m = Read-Model 'Which model for demo? (cart/jaw/phishing/all) [all]'
            Write-Host "`nRunning demo training for $m..." -ForegroundColor Green
            Invoke-Py @('train.py', '--model', $m, '--demo')
        }
        '5' { Write-Host "`nPreparing sample datasets..." -ForegroundColor Green; Invoke-Py @('train.py', '--prepare-data', 'all') }
        '6' {
            $m = Read-Model 'Which model to estimate? (cart/jaw/phishing/all) [all]'
            Invoke-Py @('train.py', '--model', $m, '--estimate-only')
        }
        '7' { Invoke-Py @('train.py', '--history') }
        '8' { Write-Host "`nStarting PHISHING training..." -ForegroundColor Green; Invoke-Py @('train.py', '--model', 'phishing') }
        { $_ -eq 'q' -or $_ -eq 'Q' } { Write-Host 'Cancelled.'; exit 0 }
        default { Write-Host 'Invalid selection.' -ForegroundColor Red; exit 1 }
    }
}

# No arguments -> wizard if interactive, else help
if ($CmdArgs.Count -eq 0) {
    if ([Environment]::UserInteractive -and -not [Console]::IsInputRedirected) { Start-Wizard } else { Show-Help }
    exit 0
}

$Cmd = [string]$CmdArgs[0]
$Rest = @()
if ($CmdArgs.Count -gt 1) { $Rest = @($CmdArgs[1..($CmdArgs.Count - 1)]) }

function Split-Model([string]$Default) {
    # Returns @{ Model = <name>; Rest = <string[] of remaining args> }
    $m = $Default
    [string[]]$r = @($script:Rest)
    if ($r.Count -gt 0 -and ($script:Models -contains $r[0].ToLower())) {
        $m = $r[0].ToLower()
        if ($r.Count -gt 1) { [string[]]$r = $r[1..($r.Count - 1)] } else { [string[]]$r = @() }
    }
    return @{ Model = $m; Rest = $r }
}

switch ($Cmd.ToLower()) {
    { @('-h', '--help', 'help') -contains $_ } {
        Show-Help
        Write-Host ''
        Write-Host '--- Underlying Python CLI options: ---' -ForegroundColor DarkGray
        Invoke-Py @('train.py', '--help')
    }
    { @('interactive', 'wizard', 'menu') -contains $_ } { Start-Wizard }
    { $Models -contains $_ } { Invoke-Py (@('train.py', '--model', $Cmd.ToLower()) + $Rest) }
    'demo' {
        $s = Split-Model 'all'
        Invoke-Py (@('train.py', '--model', $s.Model, '--demo') + $s.Rest)
    }
    { @('prepare', 'prep-data', 'sample') -contains $_ } {
        $s = Split-Model 'all'
        [string[]]$r = $s.Rest
        # Bare path (not a flag) -> treat as --data-dir
        if ($r.Count -gt 0 -and -not ([string]$r[0]).StartsWith('-')) {
            $r = @('--data-dir') + $r
        }
        Invoke-Py (@('train.py', '--prepare-data', $s.Model) + $r)
    }
    { @('download', 'dl') -contains $_ } { Invoke-Py (@('train.py', '--download-data') + $Rest) }
    'inspect' {
        $dir = 'data'
        if ($Rest.Count -gt 0) { $dir = [string]$Rest[0] }
        Invoke-Py @('-m', 'util.manager', '--inspect', $dir)
    }
    'cifake' {
        if ($Rest.Count -eq 0) {
            Write-Host 'Usage: .\train.ps1 cifake <path_to_extracted_cifake>' -ForegroundColor Red
            exit 1
        }
        Invoke-Py (@('-m', 'util.manager', '--cifake') + $Rest)
    }
    { @('dashboard', 'dash') -contains $_ } { Invoke-Py (@('train.py', '--model', 'all', '--dashboard') + $Rest) }
    { @('estimate', 'est') -contains $_ } {
        $s = Split-Model 'all'
        Invoke-Py (@('train.py', '--model', $s.Model, '--estimate-only') + $s.Rest)
    }
    { @('history', 'hist', 'logs') -contains $_ } { Invoke-Py (@('train.py', '--history') + $Rest) }
    default { Invoke-Py (@('train.py') + $CmdArgs) }
}
