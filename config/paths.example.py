# config/paths.py — local machine paths (copy this file to paths.py and fill in)
# paths.py is gitignored; paths.example.py is the tracked template.
#
# Usage:
#   cp config/paths.example.py config/paths.py
#   edit config/paths.py to match your local paths
#
# Alternatively, set environment variables:
#   LLMS_DIR    — directory containing GGUF model folders
#   PROJECT_DIR — root of this project
import os
from pathlib import Path

LLMS_DIR    = os.environ.get("LLMS_DIR",    str(Path(__file__).parent.parent.parent / "LLMmodel"))
PROJECT_DIR = os.environ.get("PROJECT_DIR", str(Path(__file__).parent.parent))
