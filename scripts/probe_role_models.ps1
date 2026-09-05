# Safe screening probe: VLM / Coder / Planner candidates
# CPU-safe: affinity pinned to cores 2+3 (mask 0xC = 50% of 4 cores), BelowNormal priority, -t 2
# Never launch llama-server without this wrapper - unrestricted launch froze the whole machine before.

param(
    [int[]]$Ctx = @(8192, 16384)
)

$Exe = "E:\Gemma4_E4B_Project\bin\llama-cpp\llama-server.exe"
$LogDir = "E:\Gemma4_E4B_Project\logs"

$Models = @(
    @{ Name = "VLM-SmolVLM-500M";   Path = "E:\LLMmodel\SmolVLM-500M-Instruct-Q8_0\SmolVLM-500M-Instruct-Q8_0.gguf"; Mmproj = "E:\LLMmodel\SmolVLM-500M-Instruct-Q8_0\mmproj-SmolVLM-500M-Instruct-Q8_0.gguf"; Role = "VLM" },
    @{ Name = "Coder-Qwen2.5-1.5B"; Path = "E:\LLMmodel\Qwen2.5-Coder-1.5B\qwen2.5-coder-1.5b-instruct-q8_0.gguf"; Mmproj = $null; Role = "Coder" },
    @{ Name = "Coder-Qwen2.5-3B";   Path = "E:\LLMmodel\Qwen2.5-Coder-3B\qwen2.5-coder-3b-instruct-q4_k_m.gguf"; Mmproj = $null; Role = "Coder" },
    @{ Name = "Planner-Qwen3-1.7B"; Path = "E:\LLMmodel\Qwen3-1.7B\Qwen3-1.7B-Q8_0.gguf"; Mmproj = $null; Role = "Planner" },
    @{ Name = "Planner-DeepSeekR1-1.5B"; Path = "E:\LLMmodel\DeepSeek-R1-1.5B\DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf"; Mmproj = $null; Role = "Planner" }
)

$Port = 8200
$Results = @()

foreach ($m in $Models) {
    foreach ($ctxSize in $Ctx) {
        $Port++
        $safeName = $m.Name -replace '[^\w\-]', '_'
        $log = Join-Path $LogDir "role_probe_${safeName}_ctx${ctxSize}.log"
        Remove-Item $log -ErrorAction SilentlyContinue

        $argList = @("-m", $m.Path, "-ngl", "999", "-c", "$ctxSize", "-t", "2", "-np", "1", "--no-webui", "--no-warmup", "--port", "$Port")
        if ($m.Mmproj) { $argList += @("--mmproj", $m.Mmproj) }

        $proc = Start-Process -FilePath $Exe -ArgumentList $argList `
            -RedirectStandardOutput $log -RedirectStandardError "$log.err" `
            -PassThru -WindowStyle Hidden

        Start-Sleep -Milliseconds 500
        try {
            $proc.ProcessorAffinity = 12   # cores 2+3 only (0xC)
            $proc.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal
        } catch {}

        $ready = $false
        $oom = $false
        for ($i = 0; $i -lt 30; $i++) {
            Start-Sleep -Seconds 2
            $content = (Get-Content $log -ErrorAction SilentlyContinue) + (Get-Content "$log.err" -ErrorAction SilentlyContinue)
            $text = $content -join "`n"
            if ($text -match "server is listening") { $ready = $true; break }
            if ($text -match "OutOfDeviceMemory|error loading model|failed to allocate") { $oom = $true; break }
            if ($proc.HasExited) { break }
        }

        $vram = (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
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
        $result = [PSCustomObject]@{
            Model = $m.Name; Role = $m.Role; Ctx = $ctxSize; Status = $status
            VRAM_MiB = $vram; DecodeTokPerSec = $decodeSpeed; VulkanBuffers = $vulkanBuf
        }
        $Results += $result
        Write-Host ("{0} | ctx={1} | {2} | VRAM={3}MiB | tok/s={4}" -f $m.Name, $ctxSize, $status, $vram, $decodeSpeed)
    }
}

$Results | Format-Table -AutoSize | Out-String -Width 300 | Write-Host
$Results | ConvertTo-Json -Depth 5 | Out-File "E:\Gemma4_E4B_Project\logs\role_probe_results.json" -Encoding utf8
