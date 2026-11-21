"""Lets scripts in scripts/ import the src package without installing it."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
