param([string]$IsaacPython)
$ErrorActionPreference='Stop'
if ($IsaacPython -and (Test-Path -LiteralPath $IsaacPython -PathType Leaf)) {
    return $IsaacPython
}
$candidates=@()
if ($env:CONDA_PREFIX) { $candidates+=Join-Path $env:CONDA_PREFIX 'python.exe' }
if ($env:USERPROFILE) {
    $candidates+=Join-Path $env:USERPROFILE 'anaconda3\envs\webot\python.exe'
    $candidates+=Join-Path $env:USERPROFILE 'miniconda3\envs\webot\python.exe'
}
foreach ($candidate in $candidates) {
    if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
}
return (Get-Command python -ErrorAction Stop).Source
