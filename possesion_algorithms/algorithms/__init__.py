"""Every module in this folder is imported; every PossessionAlgorithm subclass in it that sets its
own `name` is an algorithm run.py can run."""

import importlib
import pkgutil

from possesion_algorithms.base import PossessionAlgorithm


def _subclasses(cls):
    for sub in cls.__subclasses__():
        yield sub
        yield from _subclasses(sub)


def registry() -> dict:
    """name -> algorithm class, in name order."""
    for mod in pkgutil.iter_modules(__path__):
        importlib.import_module(f"{__name__}.{mod.name}")
    found = {}
    for cls in _subclasses(PossessionAlgorithm):
        name = cls.__dict__.get("name")
        if not name:
            continue
        if name in found and found[name] is not cls:
            raise ValueError(f"two algorithms are called {name!r}: {found[name].__qualname__} and {cls.__qualname__}")
        found[name] = cls
    return dict(sorted(found.items()))
