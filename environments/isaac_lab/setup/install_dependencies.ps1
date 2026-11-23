# Install dependencies in the isolated Isaac environment; never starts simulation or training.
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'runtime_settings.ps1')
$env:TEMP=Join-Path $runtime 'tmp'
$env:TMP=$env:TEMP
$env:PIP_CACHE_DIR=Join-Path $runtime 'pip_cache'
New-Item -ItemType Directory -Path $env:TEMP,$env:PIP_CACHE_DIR -Force | Out-Null
if (-not (Test-Path -LiteralPath $settings.isaac_python -PathType Leaf)) {
    throw 'The isolated Isaac Python is missing.'
}
& $settings.isaac_python -m pip install --no-build-isolation --index-url https://pypi.org/simple --retries 1 --timeout 20 '--only-binary=h5py,warp-lang' --requirement (Join-Path $PSScriptRoot 'runtime_requirements.txt') --editable (Join-Path $settings.isaac_lab_path 'source/isaaclab') --editable (Join-Path $settings.isaac_lab_path 'source/isaaclab_assets')
if ($LASTEXITCODE) { throw 'Isaac dependency installation failed. See the pip error above.' }
& $settings.isaac_python (Join-Path $PSScriptRoot 'configure_warp_extension.py')
if ($LASTEXITCODE) { throw 'Project Warp extension configuration failed.' }
& $settings.isaac_python -c "import h5py, warp; print('h5py:', h5py.__version__, 'Warp:', warp.__version__)"
if ($LASTEXITCODE) { throw 'Runtime dependency import validation failed.' }
Write-Host 'Isaac dependencies installed. No simulation or training was started.'
