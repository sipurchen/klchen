# config/paths.py — local machine paths (copy this file to paths.py and fill in)
# paths.py is gitignored; paths.example.py is the tracked template.
import os

LLMS_DIR    = os.environ.get("LLMS_DIR",    r"E:\LLMmodel")
PROJECT_DIR = os.environ.get("PROJECT_DIR", r"E:\Gemma4_E4B_Project")
