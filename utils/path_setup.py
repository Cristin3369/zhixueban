# -*- coding: utf-8 -*-
"""确保项目根目录在 sys.path，兼容直接运行与包导入。"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
