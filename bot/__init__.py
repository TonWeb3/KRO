"""Kronos Deriv Multipliers Bot for Gold & Silver.

Importing this package pins the Hugging Face cache INSIDE the repo, at
`model/weights/`, so the downloaded Kronos weights sit next to the code that
uses them instead of in a per-user cache under the home directory. This runs
before any submodule (and therefore before `huggingface_hub`) is imported.

Also patches huggingface_hub on Windows to gracefully fall back to zero-copy
or file copying if symlink privileges ([WinError 1314]) are not held by the user.
"""
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(ROOT, "model", "weights")

# Force HF_HOME to the repository's model/weights folder so models always save locally
os.environ["HF_HOME"] = WEIGHTS_DIR
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

os.makedirs(WEIGHTS_DIR, exist_ok=True)

# Safe symlink fallback patch for Windows
try:
    import huggingface_hub.file_download
    _orig_symlink = huggingface_hub.file_download._create_symlink

    def _safe_symlink(src, dst, new_blob=False):
        try:
            _orig_symlink(src, dst, new_blob)
        except OSError:
            # Fallback to copy on Windows when Developer Mode / symlink privileges are missing
            if os.path.islink(dst):
                os.unlink(dst)
            elif os.path.exists(dst):
                os.remove(dst)
            abs_src = os.path.normpath(os.path.join(os.path.dirname(dst), src)) if not os.path.isabs(src) else src
            shutil.copy2(abs_src, dst)

    huggingface_hub.file_download._create_symlink = _safe_symlink
except Exception:
    pass
