<#
.SYNOPSIS
  Copy recently modified files from the Windows repo to the WSL working folder.

.DESCRIPTION
  Pushes every file under the repo whose LastWriteTime is newer than the cutoff
  (default 30 minutes) to /home/monki/projects/ inside the Ubuntu-22.04 WSL
  distro. Skips the .git directory and any vendored site-packages.

.PARAMETER Minutes
  How far back to look. Default 30. Use 60 for an hour, 1440 for a day, etc.

.EXAMPLE
  .\sync-to-wsl.ps1
  .\sync-to-wsl.ps1 -Minutes 120
#>
[CmdletBinding()]
param(
    [int]$Minutes = 30
)

$ErrorActionPreference = 'Stop'

$WIN        = 'D:\Github Longterm Storage\priceprediction-noncontainer\price_predictionv3'
$WSL_DISTRO = 'Ubuntu-22.04'
$WSL_DEST   = '/home/monki/projects/'
$WIN_MOUNT  = '/mnt/d/Github Longterm Storage/priceprediction-noncontainer/price_predictionv3'

# Vendored site-packages / venv artifacts that should never be copied.
$excludeDirs = @('.git', 'ensurepip', 'lib', 'lib64', '.venv', 'node_modules', '__pycache__')

$cutoff = (Get-Date).AddMinutes(-$Minutes)

Get-ChildItem -Path $WIN -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object { $_.LastWriteTime -gt $cutoff } |
    Where-Object {
        $rel = $_.FullName.Substring($WIN.Length).TrimStart('\') -replace '\\','/'
        -not ($excludeDirs | Where-Object { $rel -like ("{0}/*" -f $_) })
    } |
    ForEach-Object {
        $rel       = $_.FullName.Substring($WIN.Length).TrimStart('\') -replace '\\','/'
        $wslPath   = "$WSL_DEST$rel"
        $wslParent = Split-Path -Path $wslPath -Parent

        if ($wslParent) {
            & wsl.exe -d $WSL_DISTRO -- sh -c "mkdir -p '$wslParent'"
        }
        & wsl.exe -d $WSL_DISTRO -- cp -f "$WIN_MOUNT/$rel" $wslPath

        Write-Host ("synced  {0}" -f $rel)
    }