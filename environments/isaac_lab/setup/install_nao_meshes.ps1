# Install geometry interactively and prepare the URDF. No simulation or training.
$ErrorActionPreference = 'Stop'
$installer = 'F:\issac\naomeshes-0.6.7-linux-x64-installer.run'
if (-not (Test-Path -LiteralPath $installer -PathType Leaf)) {
    throw "Installer missing: $installer"
}
$expected = '5F897A3328F217BAE6C0ED2A934E524C'
if ((Get-FileHash -LiteralPath $installer -Algorithm MD5).Hash -ne $expected) {
    throw 'Installer differs from the upstream 0.6.7 x64 checksum. Download it again.'
}
. (Join-Path $PSScriptRoot 'runtime_settings.ps1')
if (-not (Test-Path -LiteralPath $settings.isaac_python -PathType Leaf)) {
    throw 'Configured Isaac Python is missing.'
}
if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    throw 'WSL is unavailable. Install and initialize Ubuntu in Windows, then run this entry again.'
}
& wsl.exe --list --quiet
if ($LASTEXITCODE -ne 0) {
    throw 'Cannot access WSL. Open Ubuntu from the Windows Start menu and complete its setup. If Ubuntu already works, report the WSL error above.'
}
function Convert-ToWslPath([string] $value) {
    $full = [System.IO.Path]::GetFullPath($value)
    if ($full -notmatch '^[A-Za-z]:\\') { throw 'A local drive path is required.' }
    return '/mnt/' + $full.Substring(0,1).ToLowerInvariant() + '/' + $full.Substring(3).Replace('\','/')
}
$meshRoot = $settings.nao_meshes_path
$shellPath = Join-Path $dataRoot 'runtime\tmp\install_nao_meshes.sh'
New-Item -ItemType Directory -Path (Split-Path $shellPath -Parent) -Force | Out-Null
# Store LF line endings even if the checkout uses CRLF.
$shellText = (Get-Content -LiteralPath (Join-Path $PSScriptRoot 'install_nao_meshes.sh') -Raw).Replace("`r`n", "`n")
[System.IO.File]::WriteAllText($shellPath, $shellText, [System.Text.UTF8Encoding]::new($false))
Write-Host 'Read the upstream license and decide whether to accept it in the installer.'
& wsl.exe --exec bash (Convert-ToWslPath $shellPath) (Convert-ToWslPath $installer) (Convert-ToWslPath $meshRoot)
if ($LASTEXITCODE -ne 0) { throw 'NAO meshes installation did not complete. Configuration was not updated.' }
& $settings.isaac_python (Join-Path $PSScriptRoot 'prepare_asset.py') --urdf $settings.urdf_source --meshes $meshRoot
if ($LASTEXITCODE -ne 0) { throw 'URDF validation failed. Configuration was not updated.' }
Write-Host 'NAO geometry validated at the configured mesh path.'
