from pathlib import Path
import os

UNIDISC_DIR = Path(os.getenv("UNIDISC_DIR", Path(__file__).parent))
LIB_DIR = UNIDISC_DIR / "third_party"
CELEBV_DATA_DIR = Path(os.getenv("CELEBV_DATA_DIR", "/home/mprabhud/aswerdlo/repos/lib/CelebV-Text/downloaded_celebvtext"))
SCRATCH_CELEBV_DATA_DIR = Path("/scratch/aswerdlo/sora/celebv_text/downloaded_celebvtext")
CONFIG_PATH = os.getenv("UNIDISC_CONFIG_PATH", "configs")
HF_TOKEN = os.getenv("HF_TOKEN", os.getenv("HF_HUB_DATASETS_TOKEN"))
HF_DATASETS_CACHE = os.getenv("HF_DATASETS_CACHE", None)
HF_CACHE_DIR = os.getenv("HF_HOME", None)

if HF_CACHE_DIR is not None:
    HF_CACHE_DIR = Path(HF_CACHE_DIR)
elif HF_DATASETS_CACHE is not None:
    HF_CACHE_DIR = Path(HF_DATASETS_CACHE).parent
else:
    HF_CACHE_DIR = Path("~/.cache/huggingface").expanduser()
try:
    if SCRATCH_CELEBV_DATA_DIR.exists():
        CELEBV_DATA_DIR = SCRATCH_CELEBV_DATA_DIR
except:
    print(f"Error setting CELEBV_DATA_DIR")
