"""Per-feature on-disk store for feature activations, loaded lazily.

A full activation cache ({gate: {feature: {sequence: {"positions", "magnitudes"}}}})
for a large dictionary does not fit in memory as Python objects. The store keeps
one compact file per feature and loads features on demand, keeping recently used
ones in an LRU cache bounded by the number of (feature, sequence) entries held.

Layout of a store directory:
    store.json          format version, gates, per-feature [n_sequences, n_activations]
    sequences.pkl       list of sequence tuples; files refer to sequences by index
    metadata.pkl        provenance from the extractor (optional)
    <gate>/<feature>.npz
        seq_ids    int32  (n,)     index into sequences.pkl
        offsets    int64  (n + 1,) activations of sequence i are [offsets[i], offsets[i+1])
        positions  uint8  (total,)
        magnitudes float32 (total,)

Use open_feature_store(path) to get a mapping that behaves like the nested
dict the visualizers expect.
"""
import json
import os
import pickle
from collections import OrderedDict
from collections.abc import Mapping
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

FORMAT_VERSION = 1


class FeatureStoreWriter:
    """Write features one at a time; call close() to write the index."""

    def __init__(self, path: str, all_sequences: List[tuple], metadata: Optional[dict] = None):
        os.makedirs(path, exist_ok=True)
        self.path = path
        self.sequences = list(all_sequences)
        self.seq_ids = {seq: i for i, seq in enumerate(self.sequences)}
        if len(self.seq_ids) != len(self.sequences):
            raise ValueError("all_sequences contains duplicates")
        self.counts: Dict[str, Dict[str, List[int]]] = {}
        with open(os.path.join(path, "sequences.pkl"), "wb") as f:
            pickle.dump(self.sequences, f)
        if metadata is not None:
            with open(os.path.join(path, "metadata.pkl"), "wb") as f:
                pickle.dump(metadata, f)

    def write_feature(self, gate: str, feature: int, activations: Dict[tuple, Dict[str, list]]):
        """activations: {sequence tuple: {"positions": [...], "magnitudes": [...]}}"""
        n = len(activations)
        seq_ids = np.empty(n, dtype=np.int32)
        lengths = np.empty(n, dtype=np.int64)
        positions: List[int] = []
        magnitudes: List[float] = []
        for i, (seq, entry) in enumerate(activations.items()):
            try:
                seq_ids[i] = self.seq_ids[seq]
            except KeyError:
                raise KeyError(f"{gate}/{feature}: sequence {seq} is not in all_sequences") from None
            pos, mag = entry["positions"], entry["magnitudes"]
            if len(pos) != len(mag):
                raise ValueError(f"{gate}/{feature}: positions/magnitudes length mismatch")
            lengths[i] = len(pos)
            positions.extend(pos)
            magnitudes.extend(mag)
        offsets = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])
        pos_arr = np.asarray(positions, dtype=np.int64)
        if pos_arr.size and (pos_arr.min() < 0 or pos_arr.max() > 255):
            raise ValueError(f"{gate}/{feature}: positions outside uint8 range")
        os.makedirs(os.path.join(self.path, gate), exist_ok=True)
        np.savez(os.path.join(self.path, gate, f"{feature}.npz"), seq_ids=seq_ids, offsets=offsets,
                 positions=pos_arr.astype(np.uint8), magnitudes=np.asarray(magnitudes, dtype=np.float32))
        self.counts.setdefault(gate, {})[str(feature)] = [n, int(offsets[-1])]

    def close(self, gates: Optional[Iterable[str]] = None):
        with open(os.path.join(self.path, "store.json"), "w") as f:
            json.dump({"format_version": FORMAT_VERSION,
                       "gates": list(gates) if gates is not None else list(self.counts),
                       "n_sequences": len(self.sequences),
                       "counts": self.counts}, f)


def write_feature_store(path: str, feature_activations: Dict, all_sequences: List[tuple],
                        metadata: Optional[dict] = None):
    """Write an in-memory {gate: {feature: {sequence: {...}}}} cache as a store."""
    writer = FeatureStoreWriter(path, all_sequences, metadata)
    for gate, features in feature_activations.items():
        for feature, activations in features.items():
            writer.write_feature(gate, feature, activations)
    writer.close(gates=list(feature_activations))


class _LRU:
    """Loaded features, bounded by the total number of (feature, sequence) entries."""

    def __init__(self, max_entries: int):
        self.max_entries = max_entries
        self.items: "OrderedDict[Tuple[str, int], dict]" = OrderedDict()
        self.entries = 0

    def get(self, key):
        if key in self.items:
            self.items.move_to_end(key)
            return self.items[key]
        return None

    def put(self, key, value):
        self.items[key] = value
        self.entries += len(value)
        # Evict least recently used, but always keep the feature just loaded.
        while self.entries > self.max_entries and len(self.items) > 1:
            _, old = self.items.popitem(last=False)
            self.entries -= len(old)


class LazyGate(Mapping):
    """Features of one gate; a feature's activations load from disk on access."""

    def __init__(self, store: "LazyFeatureActivations", gate: str):
        self.store, self.gate = store, gate
        self.counts = {int(f): tuple(c) for f, c in store.index["counts"].get(gate, {}).items()}

    def __getitem__(self, feature):
        try:
            feature = int(feature)
        except (TypeError, ValueError):
            raise KeyError(feature) from None
        if feature not in self.counts:
            raise KeyError(feature)
        key = (self.gate, feature)
        cached = self.store.cache.get(key)
        if cached is None:
            cached = self.store.load(self.gate, feature)
            self.store.cache.put(key, cached)
        return cached

    def __contains__(self, feature):
        try:
            return int(feature) in self.counts
        except (TypeError, ValueError):
            return False

    def __iter__(self):
        return iter(sorted(self.counts))

    def __len__(self):
        return len(self.counts)


class LazyFeatureActivations(dict):
    """{gate: LazyGate}: drop-in for the nested feature-activation dict.

    A real dict at the top level (callers check isinstance(..., dict)); each
    gate is a LazyGate whose features load on access.
    """

    def __init__(self, path: str, max_cached_entries: int = 3_000_000):
        super().__init__()
        self.path = path
        with open(os.path.join(path, "store.json")) as f:
            self.index = json.load(f)
        if self.index.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"Unsupported feature store version {self.index.get('format_version')}")
        with open(os.path.join(path, "sequences.pkl"), "rb") as f:
            self.sequences = pickle.load(f)
        self.cache = _LRU(max_cached_entries)
        for gate in self.index["gates"]:
            self[gate] = LazyGate(self, gate)

    def load(self, gate: str, feature: int) -> Dict[tuple, Dict[str, list]]:
        data = np.load(os.path.join(self.path, gate, f"{feature}.npz"))
        seq_ids, offsets = data["seq_ids"], data["offsets"]
        positions = data["positions"].astype(np.int64).tolist()
        magnitudes = data["magnitudes"].astype(np.float64).tolist()
        sequences = self.sequences
        bounds = offsets.tolist()
        return {sequences[s]: {"positions": positions[bounds[i]:bounds[i + 1]],
                               "magnitudes": magnitudes[bounds[i]:bounds[i + 1]]}
                for i, s in enumerate(seq_ids.tolist())}


def is_feature_store(path: str) -> bool:
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "store.json"))


def open_feature_store(path: str, max_cached_entries: int = 3_000_000
                       ) -> Tuple[LazyFeatureActivations, List[tuple], Optional[dict]]:
    """Returns (feature_activations mapping, all_sequences, extractor metadata or None)."""
    store = LazyFeatureActivations(path, max_cached_entries)
    metadata = None
    meta_path = os.path.join(path, "metadata.pkl")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)
    return store, store.sequences, metadata
