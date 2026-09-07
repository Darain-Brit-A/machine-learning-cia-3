"""
Server-specific configuration module that inherits and extends root config.
"""

import sys
from pathlib import Path

# Add root directory to sys.path to ensure importing config.py works from anywhere
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config import *

# Define backend specific settings
DATABASE_URL = f"sqlite:///{DATABASE}"

# Override reload flag for server if needed
API_RELOAD = True
