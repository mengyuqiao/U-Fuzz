from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from uuid import uuid4


class Metadata(dict):
    def model_dump(self, **kwargs):
        return deepcopy(dict(self))


class Item:
    def __init__(self, memory, metadata, identifier=None):
        self.id = identifier or str(uuid4())
        self.memory = memory
        self.metadata = Metadata(deepcopy(dict(metadata)))


def item_factory(memory, metadata):
    return Item(memory, metadata)


class GeneralTextDouble:
    extract_calls = 0

    def __init__(self, _config=None):
        self.items = {}
        self.closed = False
        self.vector_db = SimpleNamespace(client=SimpleNamespace(close=lambda: None))

    def add(self, items):
        for item in items:
            self.items[item.id] = deepcopy(item)

    def get_all(self):
        return [deepcopy(item) for item in self.items.values()]

    def get(self, identifier):
        item = self.items.get(identifier)
        return deepcopy(item) if item is not None else None

    def search(self, query, top_k):
        terms = set(query.casefold().split())
        ranked = sorted(
            self.items.values(),
            key=lambda item: (-len(terms.intersection(item.memory.casefold().split())), item.id),
        )
        return [deepcopy(item) for item in ranked[:top_k]]

    def update(self, identifier, item):
        if identifier not in self.items:
            raise KeyError(identifier)
        replacement = deepcopy(item)
        replacement.id = identifier
        self.items[identifier] = replacement

    def delete(self, identifiers):
        for identifier in identifiers:
            self.items.pop(identifier, None)

    def delete_all(self):
        self.items.clear()

    def extract(self, *args, **kwargs):
        type(self).extract_calls += 1
        raise AssertionError("extract must remain inert")

    def close(self):
        self.closed = True
