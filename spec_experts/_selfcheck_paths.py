"""
Desensitization self-check.
Scans all git-tracked text files for remaining absolute Windows drive paths.
Exits 0 if clean, 1 if violations found.
"""
import subprocess, re, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Real drive-path pattern: uppercase letter + colon + backslash/slash
# Exclude known false positives in string literals (\n \r \t etc.)
DRIVE_RE = re.compile(r'(?<![a-zA-Z])([A-Z]):[/\\](?!n\b|r\b|t\b|0\b)')

# Allowed exceptions (patterns that are OK to keep)
ALLOWED_RE = re.compile(
    r'C:\\Windows'            # system path
    r'|C:\\Program Files'     # system path
    r'|\$env:USERPROFILE'     # already env-var
    r'|HKLM:|HKCU:'           # registry hives
    r'|[A-Z]:\\n'             # literal \n in string
    r'|[A-Z]:\\r'             # literal \r
    r'|[A-Z]:\\t'             # literal \t
)

TEXT_EXTS = {'.py', '.ps1', '.sh', '.md', '.mjs', '.txt', '.json', '.yaml', '.yml'}
SKIP_DIRS = {'.git', 'node_modules', '__pycache__', 'ollama_models', 'blobs'}

def get_tracked_files():
    out = subprocess.check_output(['git', 'ls-files'], cwd=ROOT, text=True)
    return [ROOT / f.strip() for f in out.splitlines() if f.strip()]

violations = []
checked = 0

for path in get_tracked_files():
    if any(part in SKIP_DIRS for part in path.parts):
        continue
    if path.suffix.lower() not in TEXT_EXTS:
        continue
    try:
        content = path.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        continue
    checked += 1
    for i, line in enumerate(content.splitlines(), 1):
        for m in DRIVE_RE.finditer(line):
            snippet = line[max(0, m.start()-5):m.start()+40].strip()
            if not ALLOWED_RE.search(snippet):
                violations.append((str(path.relative_to(ROOT)), i, snippet))

print(f"Checked {checked} files.\n")

if violations:
    print(f"FAIL — {len(violations)} violation(s) found:\n")
    for file, lineno, snippet in violations:
        print(f"  {file}:{lineno}  →  {snippet!r}")
    sys.exit(1)
else:
    print("ALL GREEN — no absolute drive paths in tracked files.")
    sys.exit(0)
