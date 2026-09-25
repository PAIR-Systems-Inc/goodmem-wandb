"""GoodMem for Weights & Biases Weave.

Retrieval from GoodMem, traced as Weave ops and evaluable with
``weave.Evaluation``::

    import weave
    from goodmem_wandb import GoodMemRetriever

    weave.init("my-project")
    retriever = GoodMemRetriever(space_name="docs")
    result = retriever.search("how do I rotate a key?")

Credentials come from ``GOODMEM_BASE_URL`` / ``GOODMEM_API_KEY`` or from
constructor keywords, and are deliberately not Weave fields: Weave publishes a
published object's pydantic fields verbatim to the trace server.
"""

from goodmem_wandb._connection import GoodMemConnection
from goodmem_wandb._spaces import GoodMemSpaceError
from goodmem_wandb.model import GoodMemRetrievalModel
from goodmem_wandb.retriever import GoodMemRetriever
from goodmem_wandb.scorers import MRR, FactRecall, RecallAtK, RetrievalHealth

__version__ = "0.2.1"

__all__ = [
    "MRR",
    "FactRecall",
    "GoodMemConnection",
    "GoodMemRetrievalModel",
    "GoodMemRetriever",
    "GoodMemSpaceError",
    "RecallAtK",
    "RetrievalHealth",
    "__version__",
]
