# v2 corpus generation (docs/expert_affinity_corpus_design.md §3.3, §4.1, §8.2).
# Renders prompts with the model's REAL chat-template tokens (<|turn>/<turn|>/<|think|>/<|channel>/<channel|>),
# NOT the v1 <start_of_turn>/<end_of_turn> (confirmed not in this model's vocabulary).
# One session = one accumulated causal transcript (thoughts retained in history, §3.4).
# CPU-safe: cores 2+3 affinity, BelowNormal, -t 2.
# ngl=0 + --swa-full for GENERATION ONLY: this SWA model's default windowed SWA cache makes
# cache_prompt silently fall back to "forcing full prompt re-processing" every turn (confirmed
# this session), which would make an 8-14 turn session cost hours. --swa-full fixes caching but
# needs the full ctx-sized SWA KV cache; that only fits VRAM at ngl=0 (tested: ngl=9 OOMs on the
# compute buffer, ngl=0 reaches "server is listening" with ~1.1GB compute buffer). Generation speed
# is secondary here (profiling itself, via imatrix, never uses this server or --swa-full).

param(
    [Parameter(Mandatory = $true)][string]$Family,
    [Parameter(Mandatory = $true)][string]$Session,
    [string]$TurnsFile = "",
    [int]$Seed = -1,
    [int]$TargetTokens = 8000,
    [int]$MaxTokens = 8191,
    [int]$Port = 8260
)

$Exe = "E:\Gemma4_E4B_Project\bin\llama-cpp\llama-server.exe"
$Model = "E:\LLMmodel\gemma4-26B-A4B-it-GGUF\google_gemma-4-26B-A4B-it-Q4_K_M.gguf"
$CorpusRoot = "E:\Gemma4_E4B_Project\data\routing_corpus_v2"
$SessionsDir = Join-Path $CorpusRoot "sessions"
$Log = "E:\Gemma4_E4B_Project\logs\build_corpus_v2_server_$Family`_$Session.log"

if ($TurnsFile -eq "") {
    $TurnsFile = Join-Path $CorpusRoot "turns\$Family`_$Session.json"
}
if (-not (Test-Path $TurnsFile)) {
    Write-Host "Turns file not found: $TurnsFile"
    exit 1
}
$spec = Get-Content $TurnsFile -Raw | ConvertFrom-Json
if ($Seed -lt 0) { $Seed = $spec.seed }

New-Item -ItemType Directory -Force -Path $SessionsDir | Out-Null

# ---- start server (CPU-safe wrapper) ----
$argList = @(
    "-m", $Model, "-ngl", "0", "--override-tensor", "\.ffn_.*_exps\.=CPU",
    "-c", "8192", "-t", "2", "-tb", "2", "-np", "1", "-sp", "-fit", "off", "--swa-full",
    "--no-webui", "--no-warmup", "--port", "$Port"
)
Remove-Item $Log -ErrorAction SilentlyContinue
Remove-Item "$Log.err" -ErrorAction SilentlyContinue
$proc = Start-Process -FilePath $Exe -ArgumentList $argList `
    -RedirectStandardOutput $Log -RedirectStandardError "$Log.err" `
    -PassThru -WindowStyle Hidden

Start-Sleep -Milliseconds 500
try {
    $proc.ProcessorAffinity = 12
    $proc.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal
} catch {}

$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 2
    $text = ((Get-Content $Log -ErrorAction SilentlyContinue) + (Get-Content "$Log.err" -ErrorAction SilentlyContinue)) -join "`n"
    if ($text -match "server is listening") { $ready = $true; break }
    if ($proc.HasExited) { break }
}
if (-not $ready) {
    Write-Host "Server never became ready — aborting."
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    exit 1
}
Write-Host "Server ready. Generating session $Family/$Session (seed=$Seed)..."

# ---- render turn-open text (§4.1) ----
function New-Line { param($s) return $s + "`n" }

$body = ""
if ($spec.system) {
    $body += New-Line "<|turn>system"
    $body += New-Line "<|think|>"
    $body += New-Line "$($spec.system)<turn|>"
} else {
    $body += New-Line "<|turn>system"
    $body += New-Line "<|think|>"
    $body += New-Line "<turn|>"
}

$turnsMeta = @()
$turnIdx = 0
$curSeed = $Seed
$accepted = $true

foreach ($turn in $spec.turns) {
    $turnIdx++
    $userStart = $body.Length
    $body += New-Line "<|turn>user"
    $body += New-Line "$($turn.user)<turn|>"
    $userEnd = $body.Length
    $body += New-Line "<|turn>model"
    $modelStart = $body.Length

    $attempt = 1
    $stopType = "unknown"
    $answer = ""
    $predictedN = 0
    $genTokens = $null
    while ($attempt -le 2) {
        $reqBody = @{
            prompt        = $body
            n_predict     = 3072
            cache_prompt  = $true
            seed          = $curSeed
            temperature   = 1.0
            top_k         = 64
            top_p         = 0.95
            return_tokens = $true
        } | ConvertTo-Json -Compress
        try {
            $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/completion" -Method Post -Body $reqBody -ContentType "application/json" -TimeoutSec 2400
            $answer = $resp.content
            $stopType = $resp.stop_type
            $predictedN = $resp.tokens_predicted
            $genTokens = $resp.tokens
        } catch {
            Write-Host "  [$Family/$Session turn $turnIdx] request failed: $($_.Exception.Message)"
            $stopType = "error"
        }
        if ($stopType -eq "eos") { break }
        Write-Host "  [$Family/$Session turn $turnIdx] stop_type=$stopType, retrying with seed+1"
        $curSeed++
        $attempt++
    }

    if ($stopType -ne "eos") {
        Write-Host "  [$Family/$Session turn $turnIdx] never got eos after 2 attempts — truncating session before this turn."
        $body = $body.Substring(0, $userStart)
        $accepted = $false
        break
    }

    # -sp renders <turn|> (control/EOG token) in content; avoid doubling it.
    if ($answer.TrimEnd() -notmatch "<turn\|>$") {
        $answer = $answer.TrimEnd() + "<turn|>"
    }
    $body += $answer
    $modelEnd = $body.Length
    $body += "`n"

    $turnsMeta += [PSCustomObject]@{
        turn         = $turnIdx
        label        = $turn.label
        user_start   = $userStart
        user_end     = $userEnd
        model_start  = $modelStart
        model_end    = $modelEnd
        stop_type    = $stopType
        tokens_predicted = $predictedN
        seed_used    = $curSeed
        gen_tokens   = $genTokens
    }

    Write-Host "  [$Family/$Session turn $turnIdx/$($spec.turns.Count)] ok ($predictedN tokens, stop=$stopType)"

    if ($modelEnd -gt ($MaxTokens * 3)) {
        # rough char->token safety valve (≈3 chars/token floor); exact accounting happens in segment_chunk_types.py
        Write-Host "  approaching MaxTokens budget by char-length heuristic — stopping session here."
        break
    }
}

$bodyPath = Join-Path $SessionsDir "$Family`_$Session.body.txt"
$turnsPath = Join-Path $SessionsDir "$Family`_$Session.turns.jsonl"
# PowerShell 5.1's "-Encoding utf8" always writes a BOM, which would corrupt the first token when this
# file is later fed to llama-tokenize.exe / imatrix. Write via .NET with UTF8Encoding($false) instead.
$utf8NoBom = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllText($bodyPath, $body, $utf8NoBom)
$turnsJsonl = ($turnsMeta | ForEach-Object { $_ | ConvertTo-Json -Compress }) -join "`n"
[System.IO.File]::WriteAllText($turnsPath, $turnsJsonl, $utf8NoBom)

Write-Host "Wrote $bodyPath ($($body.Length) chars) and $turnsPath ($($turnsMeta.Count) turns, accepted=$accepted)"

Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
Write-Host "Session build complete."
