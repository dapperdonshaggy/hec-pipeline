"""HEC Pipeline — thin wrapper that delegates to run_all.main."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from run_all import main  # noqa: E402


if __name__ == "__main__":
    main()