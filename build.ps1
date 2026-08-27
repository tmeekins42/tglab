# build.ps1 — configure, build, and (by default) run the tests.
#
# Exists mainly for one reason: a running tglab.exe holds a lock on itself, so
# the link step fails with LNK1104 and nothing else in the build reports a
# problem. Everything compiles, ctest passes against the OLD binary, and the
# change silently is not in the executable being tested. That failure mode has
# cost several rounds of "I closed it, try again".
#
# So this closes the app first. The assumption, stated so it can be argued with:
# if a build is being run, the previous session is over and an open viewer is
# stale rather than precious. -KeepApp opts out.
#
#   .\build.ps1                 configure, build, test
#   .\build.ps1 -NoTest         build only
#   .\build.ps1 -Target tglab   build one target
#   .\build.ps1 -KeepApp        do not close a running tglab.exe
#   .\build.ps1 -Fresh          delete the build directory first

[CmdletBinding()]
param(
    [string] $Config = 'Release',
    [string] $Target = '',
    [switch] $NoTest,
    [switch] $KeepApp,
    [switch] $Fresh
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$build = Join-Path $root 'build'

$cmake = 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
$ctest = 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe'
if (-not (Test-Path $cmake)) {
    $c = Get-Command cmake -ErrorAction SilentlyContinue
    if ($c) { $cmake = $c.Source } else { throw "cmake not found" }
}
if (-not (Test-Path $ctest)) {
    $c = Get-Command ctest -ErrorAction SilentlyContinue
    if ($c) { $ctest = $c.Source }
}

# --- close anything holding the output binaries -----------------------------
#
# Only our own executables, matched by path rather than by name, so a different
# tglab somewhere else on the machine is left alone. dbg_hang is included: it
# links the whole app and can be left running by an interrupted test.
if (-not $KeepApp) {
    $ours = @('tglab', 'dbg_hang')
    $killed = @()
    foreach ($n in $ours) {
        Get-Process -Name $n -ErrorAction SilentlyContinue | ForEach-Object {
            $p = $null
            try { $p = $_.Path } catch { }
            if ($p -and $p.StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) {
                try {
                    $_.CloseMainWindow() | Out-Null
                    if (-not $_.WaitForExit(2000)) { $_.Kill(); $_.WaitForExit(3000) }
                    $killed += "$($_.ProcessName) (pid $($_.Id))"
                } catch {
                    Write-Warning "could not close $($_.ProcessName) (pid $($_.Id)): $_"
                }
            }
        }
    }
    if ($killed.Count) { Write-Host "closed: $($killed -join ', ')" -ForegroundColor DarkYellow }
}

if ($Fresh -and (Test-Path $build)) {
    Write-Host "removing $build" -ForegroundColor DarkYellow
    Remove-Item -Recurse -Force $build
}

# --- configure and build ----------------------------------------------------
& $cmake -S $root -B $build | Out-Null
if ($LASTEXITCODE -ne 0) { throw "configure failed" }

$args = @('--build', $build, '--config', $Config)
if ($Target) { $args += @('--target', $Target) }

$out = & $cmake @args 2>&1
$failed = $LASTEXITCODE -ne 0

# Show errors and real warnings; the rest is noise. LNK1104 gets its own note,
# because it almost always means something is still holding the binary.
$out | Select-String -Pattern 'error|LNK[0-9]{4}|warning C4' | ForEach-Object {
    Write-Host $_ -ForegroundColor Red
}
if ($out -match 'LNK1104') {
    Write-Host "LNK1104: something still holds the output. Run with -KeepApp off, or close it by hand." -ForegroundColor Red
}
if ($failed) { throw "build failed" }
Write-Host "build ok ($Config)" -ForegroundColor Green

# --- test -------------------------------------------------------------------
if (-not $NoTest -and -not $Target -and $ctest) {
    Push-Location $build
    try {
        $t = & $ctest -C $Config 2>&1
        $t | Select-Object -Last 4 | ForEach-Object { Write-Host $_ }
        if ($LASTEXITCODE -ne 0) {
            $t | Select-String -Pattern 'Failed|\*\*\*' | ForEach-Object {
                Write-Host $_ -ForegroundColor Red
            }
            throw "tests failed"
        }
    } finally { Pop-Location }
}
