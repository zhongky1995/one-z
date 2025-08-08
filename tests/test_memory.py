import numpy as np
from memory import MemoryStore

def test_memory_store_retrieval():
    store = MemoryStore()
    store.add('first', [1.0, 0.0])
    store.add('second', [0.0, 1.0])
    results = store.search([1.0, 0.0], top_k=1)
    assert results == ['first']
