param([string] $Run)
$ErrorActionPreference = 'Stop'
$runs = Join-Path $PSScriptRoot '../experiments/poor_performance/omni/runs'
if ($Run) {
    $folder = Get-Item -LiteralPath $Run
} else {
    $models = @(Get-ChildItem -LiteralPath $runs -Directory -Filter 'train_*')
    foreach ($group in Get-ChildItem -LiteralPath $runs -Directory -Filter 'config_*') {
        $models += Get-ChildItem -LiteralPath $group.FullName -Directory -Filter 'train_*'
    }
    $folder = $models | Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName 'status.json') } |
        Sort-Object { (Get-Item -LiteralPath (Join-Path $_.FullName 'status.json')).LastWriteTimeUtc } -Descending |
        Select-Object -First 1
}
$root = (Get-Item $runs).FullName
if (-not $folder -or ($folder.Parent.FullName -ne $root -and
    ($folder.Parent.Name -notlike 'config_*' -or $folder.Parent.Parent.FullName -ne $root))) {
    throw 'Select an Isaac omni training run.'
}
$training = Join-Path $folder.FullName 'training.csv'
$statusPath = Join-Path $folder.FullName 'status.json'
$report = Join-Path $folder.FullName 'monitor_20min.jsonl'
$previous = $null
$interval = [TimeSpan]::FromMinutes(20)
$deadline = [DateTimeOffset]::Now

function Get-StopEvidence {
    $record = [ordered]@{}
    $launchTime = $folder.CreationTime
    $launcher = Get-ChildItem (Join-Path $runs 'launcher_logs') -Filter '*.out.log' |
        Where-Object { [math]::Abs(($_.CreationTime - $launchTime).TotalSeconds) -lt 45 } |
        Sort-Object { [math]::Abs(($_.CreationTime - $launchTime).TotalSeconds) } | Select-Object -First 1
    if ($launcher) {
        $prefix = $launcher.FullName -replace '\.out\.log$',''
        $record.launcher_log = $launcher.FullName
        foreach ($suffix in 'err.log','exit.json') {
            $file = "$prefix.$suffix"
            if (Test-Path -LiteralPath $file) { $record[$suffix] = @(Get-Content -LiteralPath $file -Tail 12) }
        }
    }
    $kit = Get-ChildItem (Join-Path $PSScriptRoot '../runtime/kit_state_omni/logs') -Recurse -Filter 'kit_*.log' |
        Where-Object { [math]::Abs(($_.CreationTime - $launchTime).TotalSeconds) -lt 90 } |
        Sort-Object { [math]::Abs(($_.CreationTime - $launchTime).TotalSeconds) } | Select-Object -First 1
    if ($kit) {
        $record.kit_log = $kit.FullName
        $record.kit_tail = @(Get-Content -LiteralPath $kit.FullName -Tail 10)
    }
    $retry = Join-Path $folder.FullName 'reset_retries.jsonl'
    if (Test-Path $retry) { $record.reset_tail = @(Get-Content $retry -Tail 3) }
    $record.last_checkpoint = (Get-ChildItem (Join-Path $folder.FullName 'checkpoints/*.pth') -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1).Name
    foreach ($log in 'System','Application') {
        try {
            $record[$log] = @(Get-WinEvent -FilterHashtable @{
                LogName=$log; StartTime=(Get-Date).AddMinutes(-22); Level=1,2,3
            } -ErrorAction Stop | Where-Object ProviderName -match 'Display|nvlddmkm|Application Error|Windows Error Reporting|Kernel-Power|Resource-Exhaustion' |
                Select-Object -First 10 TimeCreated,ProviderName,Id,Message)
        } catch { $record[$log] = $_.Exception.Message }
    }
    return $record
}

# A separate, exclusive handle prevents two observers writing to the same run.
$lock = [IO.File]::Open((Join-Path $folder.FullName 'monitor_20min.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
try {
    while ($true) {
        $sample = [ordered]@{time=[DateTimeOffset]::Now.ToString('o'); run=$folder.Name}
        try {
            $status = Get-Content $statusPath -Raw | ConvertFrom-Json
            $sample.status = $status.status
            if ($status.error) { $sample.error = $status.error }
            if (Test-Path $training) {
                $header = Get-Content $training -TotalCount 1
                $row = @($header; Get-Content $training -Tail 1) | ConvertFrom-Csv
                if ($null -eq $row -or $row.step -notmatch '^\d+$' -or $row.updates -notmatch '^\d+$') {
                    throw 'Last training row is unavailable or incomplete.'
                }
                $sample.step = [int]$row.step
                $sample.updates = [int]$row.updates
                $sample.last_write = (Get-Item $training).LastWriteTime.ToString('o')
            }
            if ($sample.status -in 'failed','completed') { $sample.result = $sample.status }
            elseif ($null -eq $previous) { $sample.result = 'baseline' }
            elseif ($sample.step -gt $previous.step -and $sample.updates -gt $previous.updates -and
                    (Get-Date) - (Get-Item $training).LastWriteTime -lt $interval) { $sample.result = 'advancing' }
            else { $sample.result = 'check_stall' }
            if ($sample.result -in 'check_stall','failed') { $sample.evidence = Get-StopEvidence }
            $previous = $sample
        } catch { $sample.result = 'read_error'; $sample.error = $_.Exception.Message }
        $line = $sample | ConvertTo-Json -Compress -Depth 5
        Add-Content -LiteralPath $report -Value $line -Encoding utf8
        Write-Output $line
        if ($sample.result -in 'failed','completed') { break }
        $deadline = $deadline.Add($interval)
        if ($deadline -le [DateTimeOffset]::Now) { $deadline = [DateTimeOffset]::Now.Add($interval) }
        Start-Sleep -Seconds ([int][math]::Ceiling(($deadline - [DateTimeOffset]::Now).TotalSeconds))
    }
} finally { $lock.Dispose() }
