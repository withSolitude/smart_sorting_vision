$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$outputDirectory = Join-Path $PSScriptRoot '.build'
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
$outputPath = Join-Path $outputDirectory 'host_tests.exe'
$sources = @(
    (Join-Path $PSScriptRoot 'host_tests.cpp'),
    (Join-Path $projectRoot 'User_File/2_Device/Motor/Motor_EmmV5/dvc_motor_emm_v5.cpp'),
    (Join-Path $projectRoot 'User_File/3_Robot/xy_motion.cpp')
)
& g++ -std=c++17 -Wall -Wextra -Werror -DEMM_V5_HOST_TEST -DXY_TEST_CALIBRATED `
    "-I$(Join-Path $PSScriptRoot 'host')" "-I$(Join-Path $projectRoot 'User_File')" `
    @sources -o $outputPath
if ($LASTEXITCODE -ne 0) { throw 'Host test compilation failed.' }
& $outputPath
if ($LASTEXITCODE -ne 0) { throw 'Host regression failed.' }

$lockedOutputPath = Join-Path $outputDirectory 'config_lock_test.exe'
& g++ -std=c++17 -Wall -Wextra -Werror -DEMM_V5_HOST_TEST `
    "-I$(Join-Path $PSScriptRoot 'host')" "-I$(Join-Path $projectRoot 'User_File')" `
    @sources -o $lockedOutputPath
if ($LASTEXITCODE -ne 0) { throw 'Configuration interlock test compilation failed.' }
& $lockedOutputPath
if ($LASTEXITCODE -ne 0) { throw 'Configuration interlock regression failed.' }

$integrationOutputPath = Join-Path $outputDirectory 'integration_tests.exe'
$integrationSources = @(
    (Join-Path $PSScriptRoot 'integration_tests.cpp'),
    (Join-Path $projectRoot 'User_File/3_Robot/holding_jaw.cpp'),
    (Join-Path $projectRoot 'User_File/3_Robot/sorting_storage.cpp')
)
& g++ -std=c++17 -Wall -Wextra -Werror -DEMM_V5_HOST_TEST -DXY_TEST_CALIBRATED `
    "-I$(Join-Path $PSScriptRoot 'integration_support')" `
    "-I$(Join-Path $PSScriptRoot 'host')" "-I$(Join-Path $projectRoot 'User_File')" `
    @integrationSources -o $integrationOutputPath
if ($LASTEXITCODE -ne 0) { throw 'Upper-layer integration test compilation failed.' }
& $integrationOutputPath
if ($LASTEXITCODE -ne 0) { throw 'Upper-layer integration regression failed.' }

$handshakeOutputPath = Join-Path $outputDirectory 'handshake_tests.exe'
$handshakeSources = @(
    (Join-Path $PSScriptRoot 'handshake_tests.cpp'),
    (Join-Path $projectRoot 'User_File/4_Interaction/ita_sorting_handshake.cpp')
)
& g++ -std=c++17 -Wall -Wextra -Werror -DEMM_V5_HOST_TEST -DXY_TEST_CALIBRATED `
    "-I$(Join-Path $PSScriptRoot 'integration_support')" `
    "-I$(Join-Path $PSScriptRoot 'host')" "-I$(Join-Path $projectRoot 'User_File')" `
    @handshakeSources -o $handshakeOutputPath
if ($LASTEXITCODE -ne 0) { throw 'Handshake test compilation failed.' }
& $handshakeOutputPath
if ($LASTEXITCODE -ne 0) { throw 'Handshake regression failed.' }

$robotOutputPath = Join-Path $outputDirectory 'robot_tests.exe'
$robotSources = @(
    (Join-Path $PSScriptRoot 'robot_tests.cpp'),
    (Join-Path $projectRoot 'User_File/3_Robot/holding_jaw.cpp'),
    (Join-Path $projectRoot 'User_File/3_Robot/sorting_storage.cpp'),
    (Join-Path $projectRoot 'User_File/4_Interaction/ita_sorting_handshake.cpp'),
    (Join-Path $projectRoot 'User_File/4_Interaction/ita_robot.cpp')
)
& g++ -std=c++17 -Wall -Wextra -Werror -DEMM_V5_HOST_TEST -DXY_TEST_CALIBRATED `
    "-I$(Join-Path $PSScriptRoot 'integration_support')" `
    "-I$(Join-Path $PSScriptRoot 'host')" "-I$(Join-Path $projectRoot 'User_File')" `
    @robotSources -o $robotOutputPath
if ($LASTEXITCODE -ne 0) { throw 'Robot scheduling test compilation failed.' }
& $robotOutputPath
if ($LASTEXITCODE -ne 0) { throw 'Robot scheduling regression failed.' }
