# Retry: CoT prompt, no --verbose (confirmed it only adds load-time noise, not runtime routing info),
# longer readiness wait + generous completion timeout. Still CPU-safe: cores 2+3, BelowNormal, -t 2.

$Exe = "E:\Gemma4_E4B_Project\bin\llama-cpp\llama-server.exe"
$Model = "E:\LLMmodel\gemma4-26B-A4B-it-GGUF\google_gemma-4-26B-A4B-it-Q4_K_M.gguf"
$Log = "E:\Gemma4_E4B_Project\logs\expert_visibility_probe2.log"
Remove-Item $Log -ErrorAction SilentlyContinue
Remove-Item "$Log.err" -ErrorAction SilentlyContinue

$argList = @(
    "-m", $Model, "-ngl", "9", "--override-tensor", "\.ffn_.*_exps\.=CPU",
    "-c", "8192", "-t", "2", "-np", "1", "--no-webui", "--no-warmup",
    "--port", "8251", "-lv", "999"
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
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Seconds 2
    $text = ((Get-Content $Log -ErrorAction SilentlyContinue) + (Get-Content "$Log.err" -ErrorAction SilentlyContinue)) -join "`n"
    if ($text -match "server is listening") { $ready = $true; break }
    if ($proc.HasExited) { break }
}
Write-Host "ready=$ready after $($i*2)s"

if ($ready) {
    try {
        $body = @{
            prompt = "<bos><start_of_turn>user`nLet's think step by step. What is 15% of 240? Then write a one-line python function for percentage.<end_of_turn>`n<start_of_turn>model`n"
            n_predict = 40
            temperature = 0.2
        } | ConvertTo-Json
        $resp = Invoke-RestMethod -Uri "http://127.0.0.1:8251/completion" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 120
        Write-Host "=== Response content ==="
        Write-Host $resp.content
        Write-Host "=== timings ==="
        Write-Host ($resp.timings | ConvertTo-Json -Compress)
    } catch {
        Write-Host "Request failed:" $_.Exception.Message
    }
} else {
    Write-Host "Server never became ready."
}

Start-Sleep -Seconds 1
Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 1

Write-Host "--- total log lines (out/err) ---"
$outLines = (Get-Content $Log -ErrorAction SilentlyContinue).Count
$errLines = (Get-Content "$Log.err" -ErrorAction SilentlyContinue).Count
Write-Host "out=$outLines err=$errLines"
Write-Host "--- any post-load runtime lines mentioning expert/routing/gate (excluding load_tensors/create_tensor) ---"
Select-String -Path $Log, "$Log.err" -Pattern "expert|routing|top_k|gate" -ErrorAction SilentlyContinue |
    Where-Object { $_.Line -notmatch "create_tensor|load_tensors|llama_model_loader|print_info" } |
    Select-Object -First 40
