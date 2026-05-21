"""goodmem_wandb — GoodMem client packaged for wandb agent integrations.

GoodMem is a memory layer for AI agents with support for semantic storage,
retrieval, and summarization. This package exposes GoodMem operations as a
Python client that can be used with any wandb agent or any other Python
application.
"""

from .client import GoodMemClient

__all__ = ("GoodMemClient", "__version__")

__version__ = "0.1.0"
