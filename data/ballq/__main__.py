"""python -m ballq ...   (or  python ballq_run.py ...  from the folder above)

Installs its own packages into the user profile on first run, like match_viewer.py does, so
nothing has to be pip installed by hand."""
import importlib
import importlib.util
import subprocess
import sys

REQUIRED = {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy", "yaml": "pyyaml", "cv2": "opencv-python"}


def ensure_packages(required=REQUIRED):
    missing = [pip for mod, pip in required.items() if importlib.util.find_spec(mod) is None]
    if not missing:
        return
    print(f"installing missing packages: {', '.join(missing)}")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--user", *missing])
    if r.returncode != 0:
        r = subprocess.run([sys.executable, "-m", "pip", "install", *missing])
    if r.returncode != 0:
        sys.exit("automatic install failed, run:  pip install --user " + " ".join(missing))
    importlib.invalidate_caches()


ensure_packages()
from .cli import main  # noqa: E402

main()
