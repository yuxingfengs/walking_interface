$dataRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../../sac_isaac_lab')).Path
$settings = Get-Content -Raw (Join-Path $PSScriptRoot 'simulator_config.json') | ConvertFrom-Json
$launchConfig = Get-Content -Raw (Join-Path $dataRoot 'config.json') | ConvertFrom-Json
$runtime = [IO.Path]::GetFullPath([IO.Path]::Combine($dataRoot, $launchConfig.runtime_root))
foreach ($key in @('asset','urdf_source','nao_meshes_path','isaac_python','isaac_sim_path','isaac_lab_path')) {
    $value = [string]$settings.$key
    if ($value -and -not [IO.Path]::IsPathRooted($value)) {
        if ($value.StartsWith('runtime/')) {
            $settings.$key = Join-Path $runtime $value.Substring(8)
        } elseif ($value.Replace('\','/').StartsWith('assets/')) {
            $settings.$key = [IO.Path]::GetFullPath((Join-Path (Split-Path $PSScriptRoot -Parent) $value))
        } else {
            $settings.$key = Join-Path $dataRoot $value
        }
    }
}
