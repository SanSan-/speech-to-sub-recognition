param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 7862,
    [switch]$Reload
)

$ErrorActionPreference = "Stop"

function Import-DotEnvUtf8 {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return
    }
    $utf8NoBom = [System.Text.UTF8Encoding]::new($false, $true)
    foreach ($line in [System.IO.File]::ReadAllLines($Path, $utf8NoBom)) {
        if ($line -notmatch '^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$') {
            continue
        }
        $name = $Matches[1]
        if ($null -ne [Environment]::GetEnvironmentVariable($name, "Process")) {
            continue
        }
        $value = $Matches[2].Trim()
        if ($value.Length -ge 2 -and
            (($value.StartsWith('"') -and $value.EndsWith('"')) -or
             ($value.StartsWith("'") -and $value.EndsWith("'")))) {
            $value = $value.Substring(1, $value.Length - 2)
        } else {
            $value = [regex]::Replace($value, '\s+#.*$', '').TrimEnd()
        }
        [Environment]::SetEnvironmentVariable($name, $value, "Process")
    }
}

$ProjectRoot = Split-Path -Parent $PSCommandPath
Import-DotEnvUtf8 -Path (Join-Path $ProjectRoot ".env")

$configuredHost = $env:WEB_HOST
if (-not $PSBoundParameters.ContainsKey("HostAddress") -and
    -not [string]::IsNullOrWhiteSpace($configuredHost)) {
    $HostAddress = $configuredHost.Trim()
}
$configuredPort = $env:WEB_PORT
if (-not $PSBoundParameters.ContainsKey("Port") -and
    -not [string]::IsNullOrWhiteSpace($configuredPort)) {
    $parsedPort = 0
    if (-not [int]::TryParse($configuredPort.Trim(), [ref]$parsedPort)) {
        throw "WEB_PORT должен быть целым числом от 1 до 65535."
    }
    $Port = $parsedPort
}
if ($Port -lt 1 -or $Port -gt 65535) {
    throw "WEB_PORT должен быть в диапазоне от 1 до 65535."
}

$PythonPath = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$normalizedHost = if ($HostAddress.StartsWith("[") -and $HostAddress.EndsWith("]")) {
    $HostAddress.Substring(1, $HostAddress.Length - 2)
} else {
    $HostAddress
}
$parsedAddress = $null
$isLoopback = $normalizedHost -ieq "localhost"
if (-not $isLoopback -and
    [System.Net.IPAddress]::TryParse($normalizedHost, [ref]$parsedAddress)) {
    $isLoopback = [System.Net.IPAddress]::IsLoopback($parsedAddress)
}
if (-not $isLoopback) {
    throw "Локальный макет разрешает только loopback HostAddress."
}
$HostAddress = $normalizedHost
$displayHost = if ($HostAddress.Contains(":")) { "[$HostAddress]" } else { $HostAddress }
$serviceUrl = "http://${displayHost}:$Port"

if (-not (Test-Path -LiteralPath $PythonPath)) {
    throw "Python виртуального окружения не найден: $PythonPath"
}

$listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($listener) {
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri "$serviceUrl/api/health" -TimeoutSec 2
        $health = $response.Content | ConvertFrom-Json
        if ($response.StatusCode -eq 200 -and
            $health.service -eq "speech-to-sub-recognition" -and
            $health.status -in @("ok", "degraded")) {
            Write-Host "Speech to Sub уже запущен: $serviceUrl"
            exit 0
        }
    } catch {
    }
    throw "Порт $Port занят процессом PID=$($listener.OwningProcess). Укажите другой порт."
}

Set-Location -LiteralPath $ProjectRoot

$arguments = @(
    "-B",
    "-m",
    "uvicorn",
    "speech_to_sub.web.app:app",
    "--host",
    $HostAddress,
    "--port",
    $Port.ToString()
)

if ($Reload) {
    $arguments += "--reload"
}

Write-Host "Speech to Sub: $serviceUrl"
& $PythonPath @arguments
exit $LASTEXITCODE
