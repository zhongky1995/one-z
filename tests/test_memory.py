import numpy as np
from memory import MemoryStore

def test_memory_store_retrieval():
    store = MemoryStore()
    store.add('first', [1.0, 0.0], kind='prompt')
    store.add('second', [0.0, 1.0], kind='response')

    prompt_results = store.search([1.0, 0.0], top_k=1, kind='prompt')
    response_results = store.search([0.0, 1.0], top_k=1, kind='response')

    assert prompt_results == ['first']
    assert response_results == ['second']


def test_search_handles_zero_vectors():
    store = MemoryStore()
    # zero vector in store should be ignored
    store.add('zero', [0.0, 0.0], kind='prompt')
    store.add('one', [1.0, 0.0], kind='prompt')
    results = store.search([1.0, 0.0], kind='prompt')
    assert results == ['one']

    # query with zero vector returns empty list
    assert store.search([0.0, 0.0], kind='prompt') == []
