@echo off
chcp 65001 >nul
echo ========================================
echo   Codex-For URL Configuration Tool
echo ========================================
echo.

set "CONFIG_DIR=%USERPROFILE%\.codex"
set "CONFIG_FILE=%CONFIG_DIR%\config.toml"
set "TARGET_URL=https://blackaicoding.com/v1"

if not exist "%CONFIG_DIR%" (
    mkdir "%CONFIG_DIR%"
    echo Created directory: %CONFIG_DIR%
)

powershell -NoProfile -ExecutionPolicy Bypass -Command "& { $configFile = [Environment]::ExpandEnvironmentVariables('%CONFIG_FILE%'); $targetUrl = 'https://blackaicoding.com/v1'; $q = [char]34; if (Test-Path $configFile) { $backup = $configFile + '.bak.' + (Get-Date -Format 'yyyyMMddHHmmss'); Copy-Item -LiteralPath $configFile -Destination $backup -Force; Write-Host ('Backed up existing config to: ' + $backup); $content = Get-Content -LiteralPath $configFile -Raw; if ($content -match '(?m)^[ \t]*base_url[ \t]*=') { $updated = [regex]::Replace($content, '(?m)^([ \t]*base_url[ \t]*=[ \t]*)\".*\"', ('$1' + $q + $targetUrl + $q)); [System.IO.File]::WriteAllText($configFile, $updated, (New-Object System.Text.UTF8Encoding($false))); Write-Host 'Updated existing base_url.' } else { if ($content.Length -gt 0 -and -not $content.EndsWith([Environment]::NewLine)) { $content += [Environment]::NewLine }; $content += 'base_url = ' + $q + $targetUrl + $q + [Environment]::NewLine; [System.IO.File]::WriteAllText($configFile, $content, (New-Object System.Text.UTF8Encoding($false))); Write-Host 'Added base_url.' } } else { [System.IO.File]::WriteAllText($configFile, ('base_url = ' + $q + $targetUrl + $q + [Environment]::NewLine), (New-Object System.Text.UTF8Encoding($false))); Write-Host 'Created config file with base_url.' } }"

echo.
echo Done! base_url has been set to:
echo   %TARGET_URL%
echo.
echo Config file: %CONFIG_FILE%
echo.
pause
