# Plan: Use Embeddings Instead of Keyword Similarity

From review session (Jan 2026).

## Problem

Current `_keyword_similarity()` uses basic word overlap. This misses semantic similarity:
- "authentication" vs "login" → 0.0 (no word overlap)
- "error handling" vs "exception management" → 0.0

## Solution

Use sentence-transformers for semantic similarity as optional dependency.

```python
try:
    from sentence_transformers import SentenceTransformer
    _model = SentenceTransformer('all-MiniLM-L6-v2')
    _USE_EMBEDDINGS = True
except ImportError:
    _USE_EMBEDDINGS = False
```

## pyproject.toml

```toml
[project.optional-dependencies]
embeddings = ["sentence-transformers>=2.2.0", "numpy>=1.24.0"]
```

## Done Criteria

- [ ] Add optional sentence-transformers dependency
- [ ] Implement _semantic_similarity with fallback
- [ ] Update retrieve_episodes to use new similarity
- [ ] Add tests for both modes
