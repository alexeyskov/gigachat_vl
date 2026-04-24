import sys
from pathlib import Path
import importlib.util

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import src.eval

spec = importlib.util.find_spec("vlmeval")
if spec is None or spec.origin is None:
    raise ImportError("vlmeval not found")
vlmeval_file = Path(spec.origin).resolve()
vlmeval_kit_path = vlmeval_file.parent.parent

if __name__ == "__main__":
    run_py = vlmeval_kit_path / "run.py"

    if not run_py.exists():
        print("❌ Not found run.py in VLMEvalKit")
        print(f"Expected path: {run_py}")
        sys.exit(1)

    sys.argv[0] = str(run_py)
    exec(run_py.read_text(), {"__file__": str(run_py), "__name__": "__main__"})