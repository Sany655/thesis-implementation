import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")

for path in [BACKEND_DIR, ROOT_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from backend.api import app
