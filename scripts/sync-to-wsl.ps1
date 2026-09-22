<#
.SYNOPSIS
  Full sync from the Windows repo to the WSL working directory.

.DESCRIPTION
  Walks the entire repo and ensures every file exists in
  /home/monki/projects/price_predictionv3/ inside the Ubuntu-22.04 WSL distro.

  For each file:
    * If the WSL copy does not exist           -> copy it (new)
    * If the WSL copy exists and differs       -> replace it (modified)
    * If the WSL copy exists and matches       -> skip it (unchanged)

  Files are compared by SHA-256 of their content. Symlinks are skipped.
  Vendored site-packages / venv / build caches are excluded.

.PARAMETER DryRun
  If set, only report what would change without writing anything to WSL.

.PARAMETER VerboseSync
  Print every file considered (skipped as well as synced).

.EXAMPLE
  .\sync-to-wsl.ps1
  .\sync-to-wsl.ps1 -DryRun
  .\sync-to-wsl.ps1 -VerboseSync
#>
[CmdletBinding()]
param(
    [switch]$DryRun,
    [switch]$VerboseSync
)

$ErrorActionPreference = 'Stop'

$WIN        = 'D:\Github Longterm Storage\priceprediction-noncontainer\price_predictionv3'
$WSL_DISTRO = 'Ubuntu-22.04'
$WSL_DEST   = '/home/monki/projects/price_predictionv3/'
$WIN_MOUNT  = '/mnt/d/Github Longterm Storage/priceprediction-noncontainer/price_predictionv3'

# Vendored site-packages / venv artifacts that should never be copied.
$excludeDirs = @('.git', 'ensurepip', 'lib', 'lib64', '.venv', 'node_modules', '__pycache__', '.mypy_cache', '.pytest_cache')

# --- helpers ------------------------------------------------------------------

function Test-Excluded([string]$rel) {
    foreach ($dir in $excludeDirs) {
        if ($rel -eq $dir) { return $true }
        if ($rel -like ("{0}/*" -f $dir)) { return $true }
    }
    return $false
}

function Get-FileSha256([string]$path) {
    if (-not (Test-Path -LiteralPath $path)) { return $null }
    try {
        $h = (Get-FileHash -LiteralPath $path -Algorithm SHA256 -ErrorAction Stop).Hash
        return $h.ToLower()
    } catch {
        return $null
    }
}

function Get-WslFileSha256([string]$wslPath) {
    # Run sha256sum inside WSL; if the file is missing or unreadable, return $null.
    $out = & wsl.exe -d $WSL_DISTRO -- sh -c "if [ -f '$wslPath' ]; then sha256sum '$wslPath' | awk '{print $1}'; else echo MISSING; fi" 2>$null
    if (-not $out) { return $null }
    $out = ($out | Out-String).Trim()
    if ($out -eq 'MISSING' -or $out -eq '') { return $null }
    return $out.ToLower()
}

function Sync-One([string]$rel) {
    $winPath   = Join-Path $WIN $rel
    $wslPath   = "$WSL_DEST$rel"
    $wslParent = Split-Path -Path $wslPath -Parent
    $winHash   = Get-FileSha256 $winPath
    $wslHash   = Get-WslFileSha256 $wslPath

    if ($null -eq $wslHash) {
        $status = 'new'
    } elseif ($winHash -eq $wslHash) {
        $status = 'unchanged'
    } else {
        $status = 'modified'
    }

    if (-not $DryRun -and $status -ne 'unchanged') {
        if ($wslParent) {
            & wsl.exe -d $WSL_DISTRO -- sh -c "mkdir -p '$wslParent'" | Out-Null
        }
        & wsl.exe -d $WSL_DISTRO -- cp -f "$WIN_MOUNT/$rel" $wslPath | Out-Null
    }
    return $status
}

# --- main ---------------------------------------------------------------------

Write-Host "WSL sync: $WIN -> $WSL_DEST (distro: $WSL_DISTRO)"
if ($DryRun) { Write-Host "[DryRun] no files will be written to WSL" }

$sw = [System.Diagnostics.Stopwatch]::StartNew()

# Collect every regular file under the repo, skipping excluded directories and
# symlinks (we never want to follow symlinks that point outside the repo).
$files = Get-ChildItem -Path $WIN -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object { -not $_.PSIsContainer -and -not $_.Attributes.HasFlag([IO.FileAttributes]::ReparsePoint) } |
    ForEach-Object {
        $rel = $_.FullName.Substring($WIN.Length).TrimStart('\\') -replace '\\','/'
        if (Test-Excluded $rel) { return }
        [pscustomobject]@{ Rel = $rel; FullName = $_.FullName; Length = $_.Length }
    }

$total  = @($files).Count
$counts = @{ new = 0; modified = 0; unchanged = 0 }

Write-Host ("Scanning {0} files..." -f $total)

foreach ($f in $files) {
    $rel = $f.Rel
    try {
        $status = Sync-One $rel
    } catch {
        Write-Warning ("FAILED  {0}  ::  {1}" -f $rel, $_.Exception.Message)
        continue
    }
    $counts[$status]++
    if ($status -eq 'unchanged') {
        if ($VerboseSync) { Write-Host ("  skip  {0}" -f $rel) }
    } else {
        Write-Host ("  {0,-9} {1}" -f $status, $rel)
    }
}

$sw.Stop()
Write-Host ""
Write-Host ("Done in {0:N1}s. Total: {1} | new: {2} | modified: {3} | unchanged: {4}" -f `
    $sw.Elapsed.TotalSeconds, $total, $counts.new, $counts.modified, $counts.unchanged)