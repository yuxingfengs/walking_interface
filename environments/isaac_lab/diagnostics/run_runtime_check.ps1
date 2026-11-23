# Capture native stdout/stderr while keeping the normal console visible.
# This entry only runs the empty-scene check, never training.
$ErrorActionPreference = 'Stop'
$reports = Join-Path $PSScriptRoot '../reports'
New-Item -ItemType Directory -Path $reports -Force | Out-Null
$logPath = Join-Path $reports 'runtime_console_latest.log'
$writer = [System.IO.StreamWriter]::new($logPath, $false, [System.Text.UTF8Encoding]::new($false))
$checkExit = 1
try {
    $writer.AutoFlush = $true
    $writer.WriteLine('Runtime check started: ' + [DateTimeOffset]::Now.ToString('o'))
    # Windows PowerShell wraps native stderr in ErrorRecord objects. Continue
    # consuming them and use the child process exit code for the result.
    $ErrorActionPreference = 'Continue'
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot '../../../sac_isaac_lab/launch.ps1') -Mode runtime 2>&1 | ForEach-Object {
        $line = $_.ToString()
        $writer.WriteLine($line)
        Write-Host $line
    }
    $checkExit = $LASTEXITCODE
    $writer.WriteLine('Exit code: ' + $checkExit)
} finally {
    $writer.Dispose()
}
Write-Host ('Console log: ' + $logPath)
exit $checkExit
