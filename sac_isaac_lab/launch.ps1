param([ValidateSet('offline','runtime','check','train','prepare','standing','smoke')] [string] $Mode='offline',
      [ValidatePattern('^([0-9]{3}|omni|omni_step|walk_only|walk_variable|walk_lateral|walk_lateral_random_vx)$')] [string] $Experiment,
      [ValidateSet('baseline','foot_geometry','step_20ms','step_5ms','ddpg_settled')] [string] $StandingCase,
      [ValidateSet('height','timestep','ddpg')] [string] $StandingComparison='height',
      [guid] $StandingGroup,
      [string] $Resume,
      [switch] $ContinueConfiguredRun,
      [string] $CheckOutput,
      [ValidateSet('cpu','cuda:0')] [string] $SimDevice,
      [ValidateSet('cpu','cuda:0')] [string] $LearnerDevice,
      [switch] $Headless)
$ErrorActionPreference='Stop'
$dataRoot = $PSScriptRoot
if ($Resume -and $Mode -ne 'train') {
    throw '-Resume requires -Mode train'
}
if ($Mode -eq 'standing' -and -not $StandingCase) {
    & (Join-Path $dataRoot '../environments/isaac_lab/check_standing.ps1') -StandingComparison $StandingComparison -Experiment $Experiment -Headless:$Headless
    exit $LASTEXITCODE
}
. (Join-Path $PSScriptRoot '../environments/isaac_lab/setup/runtime_settings.ps1')
$experimentArgs = @()
if ($Experiment) { $experimentArgs = @('--experiment', $Experiment) }
if ($Mode -eq 'offline') {
    $projectPython = & (Join-Path $dataRoot 'source_tasks/resolve_python.ps1') -IsaacPython $settings.isaac_python
    if ($SimDevice) { $experimentArgs += @('--sim-device', $SimDevice) }
    if ($LearnerDevice) { $experimentArgs += @('--learner-device', $LearnerDevice) }
    if ($CheckOutput) { $experimentArgs += @('--output', $CheckOutput) }
    & $projectPython -u -X utf8 (Join-Path $PSScriptRoot 'runner/offline_check.py') @experimentArgs
    exit $LASTEXITCODE
}
$entry=Join-Path $PSScriptRoot 'runner/main.py'
$params=@($entry,'--mode',$Mode)
if ($SimDevice) { $params+=@('--device',$SimDevice) }
if ($LearnerDevice) { $params+=@('--learner-device',$LearnerDevice) }
$params += $experimentArgs
if ($Resume) { $params += @('--resume', $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Resume)) }
if ($ContinueConfiguredRun) { $params += '--continue-configured-run' }
if ($Mode -eq 'standing') {
    $params += @('--standing-case', $StandingCase, '--standing-group', $StandingGroup.ToString('N'))
}
if ($settings.headless -or $Headless) { $params+= '--headless' }
if ($Mode -eq 'prepare') {$params=@((Join-Path $PSScriptRoot '../environments/isaac_lab/setup/prepare_asset.py'))}
if ($Mode -eq 'prepare' -and $settings.urdf_source) { $params+=@('--urdf', $settings.urdf_source) }
if ($Mode -eq 'prepare' -and $settings.nao_meshes_path) { $params+=@('--meshes', $settings.nao_meshes_path) }
if ($Mode -eq 'runtime') {
    $runtimeDevice = if ($SimDevice) { $SimDevice } elseif ($launchConfig.sim_device) { $launchConfig.sim_device } else { $settings.device }
    $params=@((Join-Path $PSScriptRoot '../environments/isaac_lab/diagnostics/check_runtime.py'), '--device', $runtimeDevice)
    if ($launchConfig.headless -or $Headless) { $params+='--headless' }
}
if ($Mode -eq 'prepare' -and -not $settings.isaac_python -and -not $env:ISAACLAB_PATH) {
    $projectPython = & (Join-Path $dataRoot 'source_tasks/resolve_python.ps1')
    & $projectPython -u -X utf8 @params
}
elseif ($settings.isaac_python) {
    if (-not (Test-Path -LiteralPath $settings.isaac_python -PathType Leaf)) {
        throw 'Configured Isaac Python is missing. Install Isaac Sim 5.0 and Isaac Lab 2.2.1 in Python 3.11.'
    }
    if ($settings.isaac_sim_path) {
        $env:PYTHONEXE=$settings.isaac_python
        $env:PYTHONDONTWRITEBYTECODE='1'
        $env:ISAACLAB_PATH=$settings.isaac_lab_path
        Push-Location -LiteralPath $settings.isaac_lab_path
        try { & (Join-Path $settings.isaac_sim_path 'python.bat') -u -X utf8 @params }
        finally { Pop-Location }
    } else {
        & $settings.isaac_python -u -X utf8 @params
    }
}
elseif ($env:ISAACLAB_PATH) { & (Join-Path $env:ISAACLAB_PATH 'isaaclab.bat') -p @params }
else {throw 'Isaac runtime is not installed/configured. Set isaac_python or ISAACLAB_PATH; see README.md.'}
exit $LASTEXITCODE
