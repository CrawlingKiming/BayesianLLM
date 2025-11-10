"""
Local extensions to Hugging Face TRL.

This package is intentionally separate from the upstream `trl` package
to avoid shadowing or breaking imports like `from trl import PPOTrainer`.

Usage:
    from custom_trl.gpoe_trainer import GPOETrainer
"""

from .gpoe_trainer import GPOETrainer
from .collators import PreferencePairCollator

__all__ = [
    "GPOETrainer",
    "PreferencePairCollator",
]
