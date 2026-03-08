"""category_store — manages per-category BM25 indexes with lazy loading.

each category has its own:
  - manifest (chunk metadata)
  - inverted index (keyword -> doc_ids)
  - stem cache (word -> stemmed form)
  - chunk data (compressed text)

only one category is loaded at a time to minimize RAM.
"""

import json
import math
import os
import re
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import lz4.frame as lz4
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False


# ── Porter Stemmer (minimal) ──────────────────────────────────────────

def _porter_stem(word: str) -> str:
    """minimal porter stemmer for BM25 keyword matching."""
    word = word.lower().strip()
    if len(word) <= 2:
        return word

    # step 1: plurals and past tenses
    if word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("ies"):
        word = word[:-2] + "i"
    elif word.endswith("ss"):
        pass
    elif word.endswith("s"):
        word = word[:-1]

    if word.endswith("eed"):
        pass
    elif word.endswith("ed"):
        if any(c in "aeiou" for c in word[:-2]):
            word = word[:-2]
    elif word.endswith("ing"):
        if any(c in "aeiou" for c in word[:-3]):
            word = word[:-3]

    # step 2: common suffixes
    suffixes = [
        ("ational", "ate"), ("tional", "tion"), ("enci", "ence"),
        ("anci", "ance"), ("izer", "ize"), ("ation", "ate"),
        ("ator", "ate"), ("alism", "al"), ("ness", ""),
        ("ment", ""), ("ful", ""), ("ous", ""),
    ]
    for sfx, rep in suffixes:
        if word.endswith(sfx):
            stem = word[: -len(sfx)] + rep
            if len(stem) > 2:
                word = stem
            break

    return word


# ── Data Structures ───────────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id: str
    source_file: str
    text: str
    keywords: list[str]
    token_count: int
    category: str
    created_at: str = ""


@dataclass
class CategoryIndex:
    """in-memory index for a single category."""
    name: str
    chunks: dict[str, Chunk] = field(default_factory=dict)
    inverted_index: dict[str, set[str]] = field(default_factory=dict)  # keyword -> chunk_ids
    stem_cache: dict[str, str] = field(default_factory=dict)  # word -> stemmed
    stem_to_keywords: dict[str, list[str]] = field(default_factory=dict)  # stem -> [words]
    doc_lengths: dict[str, int] = field(default_factory=dict)  # chunk_id -> word_count
    avg_doc_length: float = 0.0
    total_chunks: int = 0
    total_tokens: int = 0
    loaded: bool = False


# ── Category Store ────────────────────────────────────────────────────

class CategoryStore:
    """manages 5 category indexes with lazy loading."""

    CATEGORIES = ["personal", "work", "documents", "coding", "chat_logs"]

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.indexes: dict[str, CategoryIndex] = {}
        self.active_category: Optional[str] = None

        # create category dirs
        for cat in self.CATEGORIES:
            cat_dir = self.base_dir / cat
            cat_dir.mkdir(parents=True, exist_ok=True)
            self.indexes[cat] = CategoryIndex(name=cat)

    def _manifest_path(self, category: str) -> Path:
        return self.base_dir / category / "manifest.json"

    def _storage_path(self, category: str) -> Path:
        return self.base_dir / category / "storage.json"

    def save_category(self, category: str):
        """persist a category's index to disk."""
        idx = self.indexes[category]
        if not idx.loaded and idx.total_chunks == 0:
            return

        # save manifest (chunk metadata without text)
        manifest = []
        for cid, chunk in idx.chunks.items():
            manifest.append({
                "chunk_id": cid,
                "source_file": chunk.source_file,
                "keywords": chunk.keywords,
                "token_count": chunk.token_count,
                "category": chunk.category,
                "created_at": chunk.created_at,
            })

        with open(self._manifest_path(category), "w") as f:
            json.dump(manifest, f)

        # save chunk text (could use LZ4 in production, JSON for prototype)
        storage = {cid: chunk.text for cid, chunk in idx.chunks.items()}
        with open(self._storage_path(category), "w") as f:
            json.dump(storage, f)

    def load_category(self, category: str) -> CategoryIndex:
        """load a category from disk into RAM. unloads current active."""
        if self.active_category == category and self.indexes[category].loaded:
            return self.indexes[category]

        # unload current active category to free RAM
        if self.active_category and self.active_category != category:
            self._unload_category(self.active_category)

        idx = self.indexes[category]
        manifest_path = self._manifest_path(category)
        storage_path = self._storage_path(category)

        if manifest_path.exists() and storage_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            with open(storage_path) as f:
                storage = json.load(f)

            idx.chunks.clear()
            idx.inverted_index.clear()
            idx.stem_cache.clear()
            idx.stem_to_keywords.clear()
            idx.doc_lengths.clear()

            for entry in manifest:
                cid = entry["chunk_id"]
                text = storage.get(cid, "")
                chunk = Chunk(
                    chunk_id=cid,
                    source_file=entry["source_file"],
                    text=text,
                    keywords=entry["keywords"],
                    token_count=entry["token_count"],
                    category=entry["category"],
                    created_at=entry.get("created_at", ""),
                )
                idx.chunks[cid] = chunk
                idx.doc_lengths[cid] = len(text.split())

                # build inverted index
                for kw in chunk.keywords:
                    if kw not in idx.inverted_index:
                        idx.inverted_index[kw] = set()
                    idx.inverted_index[kw].add(cid)

            # build stem index
            for kw in idx.inverted_index:
                stemmed = _porter_stem(kw)
                idx.stem_cache[kw] = stemmed
                if stemmed not in idx.stem_to_keywords:
                    idx.stem_to_keywords[stemmed] = []
                idx.stem_to_keywords[stemmed].append(kw)

            idx.total_chunks = len(idx.chunks)
            idx.total_tokens = sum(c.token_count for c in idx.chunks.values())
            if idx.doc_lengths:
                idx.avg_doc_length = sum(idx.doc_lengths.values()) / len(idx.doc_lengths)

        idx.loaded = True
        self.active_category = category
        return idx

    def _unload_category(self, category: str):
        """unload a category from RAM (save first if dirty)."""
        idx = self.indexes[category]
        if idx.loaded:
            self.save_category(category)
            idx.chunks.clear()
            idx.inverted_index.clear()
            idx.stem_cache.clear()
            idx.stem_to_keywords.clear()
            idx.doc_lengths.clear()
            idx.loaded = False

    def add_chunk(self, category: str, chunk: Chunk):
        """add a chunk to a category. loads category if needed."""
        idx = self.load_category(category)

        idx.chunks[chunk.chunk_id] = chunk
        idx.doc_lengths[chunk.chunk_id] = len(chunk.text.split())

        # update inverted index
        for kw in chunk.keywords:
            if kw not in idx.inverted_index:
                idx.inverted_index[kw] = set()
            idx.inverted_index[kw].add(chunk.chunk_id)

            # update stem index (v4.3 fix applied here too)
            if kw not in idx.stem_cache:
                stemmed = _porter_stem(kw)
                idx.stem_cache[kw] = stemmed
                if stemmed not in idx.stem_to_keywords:
                    idx.stem_to_keywords[stemmed] = []
                idx.stem_to_keywords[stemmed].append(kw)

        idx.total_chunks = len(idx.chunks)
        idx.total_tokens = sum(c.token_count for c in idx.chunks.values())
        if idx.doc_lengths:
            idx.avg_doc_length = sum(idx.doc_lengths.values()) / len(idx.doc_lengths)

    def search_bm25(self, category: str, query: str, top_k: int = 8,
                     k1: float = 1.5, b: float = 0.75) -> list[dict]:
        """BM25 TAAT search within a single category."""
        idx = self.load_category(category)
        if not idx.chunks:
            return []

        # tokenize and stem query
        query_terms = re.findall(r'\w+', query.lower())
        expanded_terms = set()
        for term in query_terms:
            stemmed = _porter_stem(term)
            # expand stem to all known keywords
            if stemmed in idx.stem_to_keywords:
                expanded_terms.update(idx.stem_to_keywords[stemmed])
            else:
                expanded_terms.add(term)

        N = len(idx.chunks)
        avgdl = idx.avg_doc_length if idx.avg_doc_length > 0 else 1.0
        scores: dict[str, float] = {}

        for term in expanded_terms:
            if term not in idx.inverted_index:
                continue
            posting = idx.inverted_index[term]
            df = len(posting)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

            for chunk_id in posting:
                dl = idx.doc_lengths.get(chunk_id, 1)
                # term frequency in this doc
                chunk_text = idx.chunks[chunk_id].text.lower()
                tf = chunk_text.count(term)
                tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
                score = idf * tf_norm
                scores[chunk_id] = scores.get(chunk_id, 0) + score

        # rank and return top-k
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for chunk_id, score in ranked:
            chunk = idx.chunks[chunk_id]
            results.append({
                "chunk_id": chunk_id,
                "score": score,
                "text": chunk.text,
                "source_file": chunk.source_file,
                "category": category,
                "token_count": chunk.token_count,
            })
        return results

    def search_all_categories(self, query: str, top_k: int = 8,
                               k1: float = 1.5, b: float = 0.75) -> list[dict]:
        """search across all categories sequentially, merge results."""
        all_results = []
        original_active = self.active_category

        for cat in self.CATEGORIES:
            results = self.search_bm25(cat, query, top_k=top_k, k1=k1, b=b)
            all_results.extend(results)

        # restore original active category
        if original_active:
            self.load_category(original_active)

        # merge and re-rank
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]

    def get_stats(self) -> dict:
        """get stats for all categories (reads manifests without loading)."""
        stats = {"categories": {}, "total_chunks": 0, "total_tokens": 0}
        for cat in self.CATEGORIES:
            manifest_path = self._manifest_path(cat)
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = json.load(f)
                chunks = len(manifest)
                tokens = sum(e.get("token_count", 0) for e in manifest)
            else:
                chunks = 0
                tokens = 0
            stats["categories"][cat] = {"chunks": chunks, "tokens": tokens}
            stats["total_chunks"] += chunks
            stats["total_tokens"] += tokens
        stats["active_category"] = self.active_category
        return stats

    def save_all(self):
        """persist all loaded categories."""
        for cat in self.CATEGORIES:
            if self.indexes[cat].loaded:
                self.save_category(cat)
