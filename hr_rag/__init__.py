"""Northwind HR Assistant -- a small, readable RAG implementation.

Read the modules in pipeline order:

    chunking.py    stage 1+2  load documents, cut them into passages
    embeddings.py  stage 3    turn passages into vectors
    index.py       stage 4    store them, wire up the retrievers
    retrieval.py   stage 5    BM25, dense, and hybrid search
    answer.py      stage 6    Claude generates a grounded, cited answer
    evaluate.py    ---        measure whether any of it actually works
"""

__version__ = "0.1.0"
