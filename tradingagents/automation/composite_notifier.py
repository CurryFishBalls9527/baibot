"""Fan-out notifier that delegates to multiple backends."""

from __future__ import annotations

from typing import Iterable, Optional


class CompositeNotifier:
    """Wraps multiple notifiers and fans out every send() call."""

    def __init__(self, notifiers: list):
        self.notifiers = notifiers

    @property
    def enabled(self) -> bool:
        return any(n.enabled for n in self.notifiers)

    @property
    def topic(self) -> str:
        for n in self.notifiers:
            if n.enabled:
                return getattr(n, "topic", "")
        return ""

    @property
    def server(self) -> str:
        for n in self.notifiers:
            if n.enabled:
                return getattr(n, "server", "")
        return ""

    def send(
        self,
        title: str,
        message: str,
        *,
        priority: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        dedupe_key: Optional[str] = None,
        click: Optional[str] = None,
    ) -> bool:
        results = []
        for n in self.notifiers:
            try:
                results.append(
                    n.send(
                        title,
                        message,
                        priority=priority,
                        tags=tags,
                        dedupe_key=dedupe_key,
                        click=click,
                    )
                )
            except Exception:
                results.append(False)
        return any(results)
