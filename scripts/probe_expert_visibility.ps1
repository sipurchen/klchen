# Low-risk test: can stock llama-server expose per-token MoE expert routing at max log verbosity?
# CPU-safe: cores 2+3 only, BelowNormal, -t 2. ngl=9 ctx=8192 (known-safe zone from prior benchmark).

$Exe = "E:\Gemma4_E4B_Project\bin\llama-cpp\llama-server.exe"
$Model = "E:\LLMmodel\gemma4-26B-A4B-it-GGUF\google_gemma-4-26B-A4B-it-Q4_K_M.gguf"
$Log = "E:\Gemma4_E4B_Project\logs\expert_visibility_probe.log"
Remove-Item $Log -ErrorAction SilentlyContinue
Remove-Item "$Log.err" -ErrorAction SilentlyContinue

$argList = @(
    "-m", $Model, "-ngl", "9", "--override-tensor", "\.ffn_.*_exps\.=CPU",
    "-c", "8192", "-t", "2", "-np", "1", "--no-webui", "--no-warmup",
    "--port", "8250", "--verbose"
)

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

if ($ready) {
    Write-Host "Server ready, sending coding-style prompt..."
    try {
        $body = @{
            prompt = "<bos><start_of_turn>user`nWrite a python function that adds two numbers.<end_of_turn>`n<start_of_turn>model`n"
            n_predict = 12
            temperature = 0.1
        } | ConvertTo-Json
        $resp = Invoke-RestMethod -Uri "http://127.0.0.1:8250/completion" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 90
        Write-Host "Response content:" $resp.content
    } catch {
        Write-Host "Request failed:" $_.Exception.Message
    }
} else {
    Write-Host "Server never became ready."
}

Start-Sleep -Seconds 2
Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 1

Write-Host "--- log line count ---"
(Get-Content $Log -ErrorAction SilentlyContinue).Count
(Get-Content "$Log.err" -ErrorAction SilentlyContinue).Count
Write-Host "--- grep for expert/routing/top_k keywords ---"
Select-String -Path $Log, "$Log.err" -Pattern "expert|routing|top_k|gate_inp" -ErrorAction SilentlyContinue | Select-Object -First 40
