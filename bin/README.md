# bin/ — llama.cpp prebuilt binaries (not tracked in git)

This project runs Gemma4 E4B via `llama-server`, a prebuilt Windows Vulkan
binary from the official [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases).
It is **not** committed to this repo — the zip + extracted `.exe`/`.dll` set
is 200MB+ of unmodified upstream build output, not project source.

## Download

Current pinned build: **b8679** (Windows, Vulkan backend, x64)

```powershell
# From <PROJECT_ROOT>, PowerShell:
Invoke-WebRequest `
  -Uri "https://github.com/ggml-org/llama.cpp/releases/download/b8679/llama-b8679-bin-win-vulkan-x64.zip" `
  -OutFile "bin\llama-b8679-vulkan.zip"

Expand-Archive -Path "bin\llama-b8679-vulkan.zip" -DestinationPath "bin\llama-cpp" -Force
```

Verify: `bin\llama-cpp\llama-server.exe --version` should run.

## Upgrading to a newer build

1. Pick a release tag from https://github.com/ggml-org/llama.cpp/releases
   (asset name pattern: `llama-b<NUMBER>-bin-win-vulkan-x64.zip`)
2. Repeat the download/extract steps above with the new tag.
3. Update the version string in [prompts/MASTER_PROMPT.md](../prompts/MASTER_PROMPT.md)
   and the badge in [README.md](../README.md) to match.

## Why Vulkan and not CUDA

This project targets a GT 1030 (Pascal, `sm_61`) which the Vulkan backend
handles without a CUDA toolkit install. If you're on a CUDA-capable card,
swap the asset name for `llama-b<NUMBER>-bin-win-cuda-cu12.4-x64.zip` instead
and adjust `-ngl`/`-ot` flags per your VRAM budget.
