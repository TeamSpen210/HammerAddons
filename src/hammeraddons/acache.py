"""Caches an expensive computation, using an async function to compute.

If another task tries retrieving while it's already being computed, the second waits for the existing
task.
"""
from collections.abc import Awaitable, Callable, Iterator

import trio


class ACache[KeyT, ValueT]:
    """Caches an expensive computation."""
    _cache: dict[KeyT, ValueT | trio.Event]

    def __init__(self) -> None:
        self._cache = {}

    def load(self, key: KeyT, value: ValueT) -> None:
        """Load in a premade value."""
        try:
            existing = self._cache[key]
        except KeyError:
            pass
        else:
            if isinstance(existing, trio.Event):
                existing.set()
        self._cache[key] = value

    def __iter__(self) -> Iterator[tuple[KeyT, ValueT]]:
        """Iterate through the currently cached items."""
        for key, value in self._cache.items():
            if not isinstance(value, trio.Event):
                yield key, value

    def __len__(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        """Remove all the contents."""
        self._cache.clear()

    async def fetch[**Args](
        self, key: KeyT, func: Callable[Args, Awaitable[ValueT]],
        /, *args: Args.args, **kwargs: Args.kwargs,
    ) -> ValueT:
        """Retreive from the cache, or compute the value."""
        while True:
            await trio.lowlevel.checkpoint()
            try:
                result = self._cache[key]
            except KeyError:
                self._cache[key] = evt = trio.Event()
                try:
                    self._cache[key] = result = await func(*args, **kwargs)
                except Exception:
                    # Undo everything.
                    if self._cache[key] is evt:
                        del self._cache[key]
                    raise
                finally:
                    evt.set()
                return result
            if isinstance(result, trio.Event):
                await result.wait()
                continue
            return result
