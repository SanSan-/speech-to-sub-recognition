param(
    [switch]$Qwen,
    [switch]$Parakeet
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot

if (-not $Qwen -and -not $Parakeet) {
    $Qwen = $true
    $Parakeet = $true
}

function Assert-LastExitCode {
    param([string]$Action)
    if ($LASTEXITCODE -ne 0) {
        throw "$Action завершилось с кодом $LASTEXITCODE."
    }
}

function Install-WorkerRuntime {
    param(
        [Parameter(Mandatory = $true)][string]$PythonVersion,
        [Parameter(Mandatory = $true)][string]$RuntimeName,
        [Parameter(Mandatory = $true)][string]$RequirementsFile,
        [Parameter(Mandatory = $true)][string]$PipVersion
    )

    $RuntimePath = Join-Path $ProjectRoot "resources\runtimes\$RuntimeName"
    $PythonPath = Join-Path $RuntimePath "Scripts\python.exe"
    $LockPath = Join-Path $ProjectRoot "requirements-workers\$RequirementsFile"

    if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
        & py "-$PythonVersion" -m venv $RuntimePath
        Assert-LastExitCode "Создание runtime $RuntimeName"
    }
    & $PythonPath -m pip install --upgrade "pip==$PipVersion"
    Assert-LastExitCode "Установка pip для $RuntimeName"
    & $PythonPath -m pip install --requirement $LockPath
    Assert-LastExitCode "Установка зависимостей $RuntimeName"
    & $PythonPath -m pip check
    Assert-LastExitCode "Проверка зависимостей $RuntimeName"
}

if ($Parakeet) {
    Install-WorkerRuntime `
        -PythonVersion "3.14" `
        -RuntimeName "parakeet" `
        -RequirementsFile "parakeet.txt" `
        -PipVersion "25.3"
}

if ($Qwen) {
    Install-WorkerRuntime `
        -PythonVersion "3.11" `
        -RuntimeName "qwen" `
        -RequirementsFile "qwen.txt" `
        -PipVersion "26.2.1"
}
