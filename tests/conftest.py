import sys
from pathlib import Path

# Add repository root to PYTHONPATH so tests can import "data", "runner", etc.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
