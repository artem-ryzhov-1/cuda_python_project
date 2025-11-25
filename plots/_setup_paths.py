from pathlib import Path
import sys

# Detect project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# CUDA paths
CUDA_INPUT = PROJECT_ROOT / "app" / "cuda" / "input"
CUDA_OUTPUT = PROJECT_ROOT / "app" / "cuda" / "output"
