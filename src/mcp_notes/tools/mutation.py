"""Coordinate source mutations with embedding-generation migration."""

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps

from mcp_notes.singletons import get_indexer


@dataclass
class _MutationScope:
    owner: object
    stack: AsyncExitStack
    ready: bool = False


_scope: ContextVar[_MutationScope | None] = ContextVar("note_mutation_scope", default=None)


async def prepare_collection_mutation() -> None:
    """Acquire readiness after validation, retaining the lock until the tool returns."""
    scope = _scope.get()
    if scope is None or scope.owner is not asyncio.current_task():
        raise RuntimeError("Collection preparation requires a tool mutation scope")
    if not scope.ready:
        indexer = await get_indexer()
        await scope.stack.enter_async_context(indexer.collection_operation())
        scope.ready = True


def collection_mutation[**P, R](
    function: Callable[P, Awaitable[R]],
) -> Callable[P, Awaitable[R]]:
    """Provide a lazy lock scope without contacting services before validation."""

    @wraps(function)
    async def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        async with AsyncExitStack() as stack:
            token = _scope.set(_MutationScope(asyncio.current_task(), stack))
            try:
                return await function(*args, **kwargs)
            finally:
                _scope.reset(token)

    return wrapped
