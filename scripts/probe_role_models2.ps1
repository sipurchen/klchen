# Round 2: reduced-ngl probes for models that OOM'd at ngl=999
$Exe = "E:\Gemma4_E4B_Project\bin\llama-cpp\llama-server.exe"
$LogDir = "E:\Gemma4_E4B_Project\logs"

$Models = @(
    @{ Name = "Coder-Qwen2.5-1.5B"; Path = "E:\LLMmodel\Qwen2.5-Coder-1.5B\qwen2.5-coder-1.5b-instruct-q8_0.gguf"; Ngls = @(25) },
    @{ Name = "Coder-Qwen2.5-3B";   Path = "E:\LLMmodel\Qwen2.5-Coder-3B\qwen2.5-coder-3b-instruct-q4_k_m.gguf"; Ngls = @(24) },
    @{ Name = "Planner-Qwen3-1.7B"; Path = "E:\LLMmodel\Qwen3-1.7B\Qwen3-1.7B-Q8_0.gguf"; Ngls = @(22) },
    @{ Name = "Planner-DeepSeekR1-1.5B"; Path = "E:\LLMmodel\DeepSeek-R1-1.5B\DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf"; Ngls = @(25) }
)

$Ctx = @(8192, 16384)
$Port = 8300
$Results = @()

foreach ($m in $Models) {
    foreach ($ngl in $m.Ngls) {
        foreach ($ctxSize in $Ctx) {
            $Port++
            $safeName = $m.Name -replace '[^\w\-]', '_'
            $log = Join-Path $LogDir "role_probe2_${safeName}_ngl${ngl}_ctx${ctxSize}.log"
            Remove-Item $log -ErrorAction SilentlyContinue
            Remove-Item "$log.err" -ErrorAction SilentlyContinue

            $argList = @("-m", $m.Path, "-ngl", "$ngl", "-c", "$ctxSize", "-t", "2", "-np", "1", "--no-webui", "--no-warmup", "--port", "$Port")

            $proc = Start-Process -FilePath $Exe -ArgumentList $argList `
                -RedirectStandardOutput $log -RedirectStandardError "$log.err" `
                -PassThru -WindowStyle Hidden

            Start-Sleep -Milliseconds 500
            try {
                $proc.ProcessorAffinity = 12
                $proc.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal
            } catch {}

            $ready = $false
            $oom = $false
            for ($i = 0; $i -lt 25; $i++) {
                Start-Sleep -Seconds 2
                $content = (Get-Content $log -ErrorAction SilentlyContinue) + (Get-Content "$log.err" -ErrorAction SilentlyContinue)
                $text = $content -join "`n"
                if ($text -match "server is listening") { $ready = $true; break }
                if ($text -match "OutOfDeviceMemory|error loading model|failed to allocate") { $oom = $true; break }
                if ($proc.HasExited) { break }
            }

            $decodeSpeed = $null
            $vulkanBuf = ($text -split "`n" | Select-String "Vulkan0.*buffer size") -join " | "

            if ($ready) {
                try {
                    $body = @{ prompt = "<|im_start|>user`nSay hi in 3 words<|im_end|>`n<|im_start|>assistant`n"; n_predict = 16; temperature = 0.1 } | ConvertTo-Json
                    $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/completion" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 60
                    $decodeSpeed = $resp.timings.predicted_per_second
                } catch {}
            }

            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
            Start-Sleep -Seconds 2

            $status = if ($ready) { "OK" } elseif ($oom) { "OOM" } else { "TIMEOUT" }
            $Results += [PSCustomObject]@{
                Model = $m.Name; Ngl = $ngl; Ctx = $ctxSize; Status = $status
                DecodeTokPerSec = $decodeSpeed; VulkanBuffers = $vulkanBuf
            }
            Write-Host ("{0} | ngl={1} | ctx={2} | {3} | tok/s={4}" -f $m.Name, $ngl, $ctxSize, $status, $decodeSpeed)
        }
    }
}
$Results | Format-Table -AutoSize | Out-String -Width 300 | Write-Host
