# v2 Tier-1 profiling driver (docs/expert_affinity_corpus_design.md §5.3, §5.4, §7, §8.2).
# Runs one imatrix invocation per job in jobs.json (each job = a D×2 prefix at one cut point,
# --chunks 1 so only the real prefix is processed, never the duplicated half).
# CPU-safe: cores 2+3 affinity, BelowNormal, -t 2. One model process at a time.
# VRAM guard: abort a job if the three Vulkan0 buffer-size lines sum > 1700 MiB.

param(
    [string]$JobsFile = "E:\Gemma4_E4B_Project\data\routing_corpus_v2\jobs.json",
    [string]$File = "",       # ad-hoc single-job mode: -File <path> -Ctx <n> -Out <path>
    [int]$Ctx = 0,
    [string]$Out = ""
)

$Exe = "E:\Gemma4_E4B_Project\bin\llama-cpp\llama-imatrix.exe"
$Model = "E:\LLMmodel\gemma4-26B-A4B-it-GGUF\google_gemma-4-26B-A4B-it-Q4_K_M.gguf"
$LogDir = "E:\Gemma4_E4B_Project\logs\imatrix_v2"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$VramLimitMiB = 1700

function Run-Job {
    param($InFile, $CtxSize, $OutFile, $JobLabel)

    $log = Join-Path $LogDir ("$JobLabel.log")
    Remove-Item $log -ErrorAction SilentlyContinue
    Remove-Item "$log.err" -ErrorAction SilentlyContinue

    $argList = @(
        "-m", $Model, "-ngl", "9", "--override-tensor", "\.ffn_.*_exps\.=CPU",
        "-c", "$CtxSize", "-b", "512", "-ub", "512", "-t", "2", "-tb", "2", "-fit", "off",
        "-f", $InFile, "--chunks", "1",
        "--parse-special", "--no-escape", "--no-ppl",
        "--output-format", "gguf", "-o", $OutFile
    )

    $proc = Start-Process -FilePath $Exe -ArgumentList $argList `
        -RedirectStandardOutput $log -RedirectStandardError "$log.err" `
        -PassThru -WindowStyle Hidden

    Start-Sleep -Milliseconds 500
    try {
        $proc.ProcessorAffinity = 12
        $proc.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal
    } catch {}

    # Per-job wall-clock ceiling: 30s + 0.12s * ctx (§7 rule 4), ~3x expected pass time.
    $ceilingSeconds = [Math]::Ceiling(30 + 0.12 * $CtxSize)
    $elapsed = 0
    $vramOk = $null
    while ($elapsed -lt $ceilingSeconds) {
        Start-Sleep -Seconds 2
        $elapsed += 2
        $text = ((Get-Content $log -ErrorAction SilentlyContinue) + (Get-Content "$log.err" -ErrorAction SilentlyContinue)) -join "`n"

        if ($vramOk -eq $null -and ($text -match "compute buffer size")) {
            $sum = 0.0
            foreach ($m in [regex]::Matches($text, "Vulkan0[^\r\n]*buffer size\s*=\s*([\d.]+)\s*MiB")) {
                $sum += [double]$m.Groups[1].Value
            }
            if ($sum -gt $VramLimitMiB) {
                Write-Host "  [$JobLabel] VRAM guard tripped: $sum MiB > $VramLimitMiB MiB — aborting job."
                Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
                return "VRAM_ABORT"
            }
            $vramOk = $sum
        }
        if ($proc.HasExited) { break }
    }
    if (-not $proc.HasExited) {
        Write-Host "  [$JobLabel] exceeded ${ceilingSeconds}s ceiling — killing (check $log for a hang)."
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        return "TIMEOUT"
    }

    $text = ((Get-Content $log -ErrorAction SilentlyContinue) + (Get-Content "$log.err" -ErrorAction SilentlyContinue)) -join "`n"
    if (-not (Test-Path $OutFile)) {
        Write-Host "  [$JobLabel] FAILED — no output file. Log tail:"
        Get-Content "$log.err" -Tail 20 -ErrorAction SilentlyContinue
        return "NO_OUTPUT"
    }
    if ($text -notmatch "computing over 1 chunks, n_ctx=$CtxSize") {
        Write-Host "  [$JobLabel] WARNING: expected log line 'computing over 1 chunks, n_ctx=$CtxSize' not found — verify manually."
        return "OK_UNVERIFIED"
    }
    return "OK"
}

$results = @()

if ($File -ne "") {
    if ($Ctx -le 0 -or $Out -eq "") {
        Write-Host "Ad-hoc mode needs -File -Ctx -Out"
        exit 1
    }
    $status = Run-Job -InFile $File -CtxSize $Ctx -OutFile $Out -JobLabel "adhoc"
    Write-Host "adhoc -> $status"
    exit 0
}

if (-not (Test-Path $JobsFile)) {
    Write-Host "Jobs file not found: $JobsFile (run scripts/build_imatrix_jobs.py first)"
    exit 1
}
$jobs = Get-Content $JobsFile -Raw | ConvertFrom-Json

foreach ($job in $jobs) {
    $label = "$($job.session)_$($job.k)"
    Write-Host "=== Job $label (ctx=$($job.ctx), type=$($job.window_type)) ==="
    $status = Run-Job -InFile $job.infile -CtxSize $job.ctx -OutFile $job.outfile -JobLabel $label
    Write-Host "  $label -> $status"
    $results += [PSCustomObject]@{ session = $job.session; k = $job.k; status = $status; outfile = $job.outfile }
    Start-Sleep -Seconds 5   # let VRAM/driver settle before the next model load (§7 rule 2)
}

$resultsPath = Join-Path $LogDir "job_results.json"
$results | ConvertTo-Json | Set-Content -Path $resultsPath -Encoding utf8
Write-Host "`n=== Done. $($results.Count) jobs. Results: $resultsPath ==="
$results | Group-Object status | ForEach-Object { Write-Host "  $($_.Name): $($_.Count)" }
