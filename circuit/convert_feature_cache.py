"""Convert a pickled feature-activation cache into a lazily loaded feature store.

The extractor (circuit/copy_find_features.py) pickles
{gate: {feature: {sequence: {"positions": [...], "magnitudes": [...]}}}}; for a
large dictionary that pickle is tens of GB of Python objects once loaded. This
streams the pickle once with a minimal opcode interpreter, builds one feature at
a time, writes it to the store (circuit/feature_store.py) and frees it, so memory
stays at roughly one feature plus the shared sequence tuples.

    python -m circuit.convert_feature_cache \
        --features features/run/run_features.p \
        --metadata features/run/run_metadata.p \
        --out features/run_store

Then point the visualizer at the store directory instead of the pickle.
"""
import argparse
import mmap
import pickle
import struct
import time

from circuit.feature_store import FeatureStoreWriter

_unpack_d = struct.Struct(">d").unpack_from
_unpack_I = struct.Struct("<I").unpack_from
_unpack_i = struct.Struct("<i").unpack_from
_unpack_H = struct.Struct("<H").unpack_from


def stream_features(path: str, on_feature, report_every: int = 1 << 30):
    """Call on_feature(gate, feature, activations) for every feature in the pickle.

    Only the opcodes the cache uses are supported. The memo keeps strings and
    tuples only (the sequence keys, which later features reference); feature
    dicts are cleared after on_feature returns.
    """
    f = open(path, "rb")
    m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    end = len(m)
    memo, memo_n = {}, 0
    stack, marks = [], []
    gate_name, current = None, None          # current = (gate, feature, dict) being built
    gates = []
    i, t0, next_report = 0, time.time(), report_every

    def flush():
        nonlocal current
        if current is not None:
            gate, feature, d = current
            on_feature(gate, feature, d)
            d.clear()
            current = None

    while i < end:
        op = m[i]; i += 1
        if op == 0x68:                                   # BINGET
            stack.append(memo.get(m[i])); i += 1
        elif op == 0x94:                                 # MEMOIZE
            top = stack[-1]
            if type(top) is tuple or type(top) is str:   # only these are referenced again
                memo[memo_n] = top
            memo_n += 1
        elif op == 0x28:                                 # MARK
            marks.append(len(stack))
        elif op == 0x5d:                                 # EMPTY_LIST
            stack.append([])
        elif op == 0x4b:                                 # BININT1
            stack.append(m[i]); i += 1
        elif op == 0x47:                                 # BINFLOAT
            stack.append(_unpack_d(m, i)[0]); i += 8
        elif op == 0x61:                                 # APPEND
            v = stack.pop(); stack[-1].append(v)
        elif op == 0x65:                                 # APPENDS
            k = marks.pop(); stack[k - 1].extend(stack[k:]); del stack[k:]
        elif op == 0x7d:                                 # EMPTY_DICT
            # Values are written in full before their SETITEMS batch, and
            # single-item dicts use SETITEM without a MARK, so a dict's role
            # comes from the key just pushed: gate names are str, features
            # int, sequences tuple. A new feature or gate means the previous
            # feature is complete.
            key = stack[-1] if stack else None
            d = {}
            if stack and type(key) is str:
                flush(); gate_name = key; gates.append(key)
            elif type(key) is int:
                flush(); current = (gate_name, key, d)
            stack.append(d)
        elif op == 0x85 or op == 0x86 or op == 0x87:     # TUPLE1/2/3
            n = op - 0x84; tup = tuple(stack[-n:]); del stack[-n:]; stack.append(tup)
        elif op == 0x74:                                 # TUPLE
            k = marks.pop(); tup = tuple(stack[k:]); del stack[k:]; stack.append(tup)
        elif op == 0x75:                                 # SETITEMS
            k = marks.pop(); d = stack[k - 1]; items = stack[k:]; del stack[k:]
            for j in range(0, len(items), 2):
                d[items[j]] = items[j + 1]
        elif op == 0x73:                                 # SETITEM
            v = stack.pop(); key = stack.pop(); stack[-1][key] = v
        elif op == 0x8c:                                 # SHORT_BINUNICODE
            n = m[i]; i += 1; stack.append(m[i:i + n].decode("utf-8")); i += n
        elif op == 0x58:                                 # BINUNICODE
            n = _unpack_I(m, i)[0]; i += 4; stack.append(m[i:i + n].decode("utf-8")); i += n
        elif op == 0x6a:                                 # LONG_BINGET
            stack.append(memo.get(_unpack_I(m, i)[0])); i += 4
        elif op == 0x4d:                                 # BININT2
            stack.append(_unpack_H(m, i)[0]); i += 2
        elif op == 0x4a:                                 # BININT
            stack.append(_unpack_i(m, i)[0]); i += 4
        elif op == 0x95:                                 # FRAME header (protocol 4)
            i += 8
        elif op == 0x80:                                 # PROTO
            i += 1
        elif op == 0x4e:                                 # NONE
            stack.append(None)
        elif op == 0x88:                                 # NEWTRUE
            stack.append(True)
        elif op == 0x89:                                 # NEWFALSE
            stack.append(False)
        elif op == 0x2e:                                 # STOP
            flush()
            break
        else:
            raise ValueError(f"Unsupported pickle opcode 0x{op:02x} at byte {i - 1}")
        if i >= next_report:
            rate = i / (time.time() - t0)
            print(f"  {i / 1e9:.1f} / {end / 1e9:.1f} GB ({rate / 1e6:.1f} MB/s, "
                  f"ETA {(end - i) / rate / 60:.0f} min)", flush=True)
            next_report += report_every
    else:
        raise ValueError("Pickle ended without STOP (truncated file?)")
    return gates


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--features", required=True, help="<run>_features.p from the extractor")
    ap.add_argument("--metadata", required=True, help="<run>_metadata.p (provides all_sequences)")
    ap.add_argument("--out", required=True, help="Output store directory")
    args = ap.parse_args()

    with open(args.metadata, "rb") as f:
        metadata = pickle.load(f)
    writer = FeatureStoreWriter(args.out, metadata["all_sequences"],
                                metadata={k: v for k, v in metadata.items() if k != "all_sequences"})
    n_written = [0]

    def on_feature(gate, feature, activations):
        writer.write_feature(gate, feature, activations)
        n_written[0] += 1

    t0 = time.time()
    gates = stream_features(args.features, on_feature)
    writer.close(gates=gates)
    print(f"Wrote {n_written[0]} features ({', '.join(f'{g}: {len(writer.counts.get(g, {}))}' for g in gates)}) "
          f"to {args.out} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
