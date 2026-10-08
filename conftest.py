"""Настройка путей для pytest: добавляем корень проекта и src в sys.path."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))