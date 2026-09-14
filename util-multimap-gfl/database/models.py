
from typing import TypedDict

class DomainMetadata(TypedDict):
    modes: dict
    predicates: list
    types: list
    target_predicate: str
    raw_modes: list

class PrologData(TypedDict):
    facts: list[str]
    pos: list[str]
    neg: list[str]
    modes: list[str]