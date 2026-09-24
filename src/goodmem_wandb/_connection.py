"""Connection settings and SDK client ownership.

Nothing in this module is ever a field on a ``weave.Object``. Weave publishes
an object's pydantic fields verbatim to the W&B trace server, and its
``should_redact`` helper does not run on that path, so a credential stored as
a field would be uploaded in plaintext. Components hold a connection on a
private attribute instead. ``tests/test_regressions.py`` asserts this.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import os
from typing import Any, cast

from goodmem import Goodmem
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from goodmem_wandb._typing import GoodmemClient

DEFAULT_TIMEOUT = 60.0


class GoodMemConnection(BaseModel):
    """How to reach a GoodMem server.

    Either inject a caller-owned ``client`` -- the connection pool and TLS
    configuration are then yours, and this class never closes it -- or supply
    ``base_url``/``api_key`` and let each operation open and close its own
    client. Both fall back to ``GOODMEM_BASE_URL`` and ``GOODMEM_API_KEY``.
    There is no process-wide client cache.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    base_url: str | None = None
    # Excluded from dumps and masked in repr. A retriever is routinely
    # rendered into a traceback, a notebook cell and an agent's state.
    api_key: SecretStr | None = Field(default=None, exclude=True, repr=False)
    verify_ssl: bool | str = True
    timeout: float = DEFAULT_TIMEOUT
    client: Any = Field(default=None, exclude=True, repr=False)

    def _resolve(self) -> tuple[str, str]:
        url = self.base_url or os.getenv("GOODMEM_BASE_URL", "")
        key = (
            self.api_key.get_secret_value()
            if self.api_key is not None
            else os.getenv("GOODMEM_API_KEY", "")
        )
        if not url:
            raise ValueError(
                "GoodMem base URL is required. Pass base_url, inject a client, "
                "or set GOODMEM_BASE_URL."
            )
        if not key:
            raise ValueError(
                "GoodMem API key is required. Pass api_key, inject a client, "
                "or set GOODMEM_API_KEY."
            )
        return url.rstrip("/"), key

    @contextmanager
    def session(self) -> Iterator[GoodmemClient]:
        """Yield an SDK client, closing it only when this object created it."""
        if self.client is not None:
            # An injected client keeps its own server, credentials and TLS
            # settings. Environment variables must not redirect it.
            yield self.client
            return

        url, key = self._resolve()
        client = Goodmem(
            base_url=url,
            api_key=key,
            verify=self.verify_ssl,
            timeout=self.timeout,
        )
        try:
            # The SDK builds its API groups at runtime, so the static type
            # does not advertise them; GoodmemClient describes what we call.
            yield cast(GoodmemClient, client)
        finally:
            client.close()


#: Constructor keywords a component forwards to its private connection rather
#: than declaring as weave fields.
CONNECTION_KEYS = ("base_url", "api_key", "verify_ssl", "timeout", "client")


def split_connection_kwargs(data: dict[str, Any]) -> GoodMemConnection:
    """Pop the connection keywords out of ``data`` and build a connection.

    ``data`` is mutated: what remains is the component's own weave fields.
    """
    return GoodMemConnection(
        **{key: data.pop(key) for key in CONNECTION_KEYS if key in data}
    )
