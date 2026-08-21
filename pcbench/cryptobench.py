"""Hardware crypto and modern compression codecs.

Two gaps the standard library cannot fill:

* **AES.** There is no AES primitive in the standard library, and a pure-Python
  implementation would never reach the AES-NI or ARMv8 crypto instructions that
  make such a benchmark meaningful — it would measure table lookups instead.
  ``cryptography`` binds to OpenSSL, which does use them.
* **Modern compression.** ``zlib`` dates from 1995. Zstandard and LZ4 are what
  contemporary software actually uses, and they behave very differently: LZ4
  trades ratio for speed, Zstandard aims for both.

Each benchmark validates its own output — a round trip that does not reproduce
the input, or a digest that does not match, is a hardware fault rather than a
slow result.
"""

from __future__ import annotations

from .core import ValidationError, clock, summarize
from .optional import have

MB = 1024 * 1024


def _payload(size: int = 8 * MB) -> bytes:
    """Deterministic, semi-compressible data — the same on every machine."""
    import random
    rnd = random.Random(31337)
    words = [b"alpha", b"bravo", b"charlie", b"delta", b"echo", b"foxtrot"]
    out = bytearray()
    while len(out) < size:
        out += b" ".join(rnd.choice(words) for _ in range(96)) + b"\n"
    return bytes(out[:size])


def available() -> dict:
    return {name: have(name) for name in
            ("cryptography", "zstandard", "lz4", "blake3")}


def _rate(fn, data_len: int, seconds: float, repeats: int) -> dict:
    rates = []
    for _ in range(max(1, repeats)):
        start = clock()
        n = 0
        while clock() - start < seconds:
            fn()
            n += 1
        elapsed = clock() - start
        rates.append(n * data_len / elapsed / MB)
    s = summarize(rates)
    return {"rate": round(s["median"], 1), "cv": round(s["cv"], 4)}


def bench_aes(seconds: float = 0.6, repeats: int = 2) -> dict:
    """AES-256-GCM throughput, in MB/s.

    GCM is the mode nearly all modern TLS and disk encryption uses. OpenSSL
    dispatches to AES-NI (x86) or the ARMv8 crypto extensions where present, so
    a machine with hardware AES is typically several times faster here — the
    difference this benchmark exists to expose.
    """
    if not have("cryptography"):
        return {"skipped": True, "error": "cryptography not installed"}
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    key = bytes(range(32))
    nonce = bytes(range(12))
    data = _payload(4 * MB)
    aes = AESGCM(key)

    # Validate before timing: a round trip that loses data is a fault.
    if aes.decrypt(nonce, aes.encrypt(nonce, data, None), None) != data:
        raise ValidationError(
            "aes: encrypt/decrypt round trip did not reproduce the input — "
            "indicates a memory or CPU fault")

    enc = _rate(lambda: aes.encrypt(nonce, data, None), len(data),
                seconds, repeats)
    return {"unit": "MB/s", "rate": enc["rate"], "cv": enc["cv"],
            "cipher": "AES-256-GCM", "validated": True,
            "note": "uses AES-NI / ARM crypto extensions where available"}


def bench_zstd(seconds: float = 0.6, repeats: int = 2) -> dict:
    """Zstandard compression throughput and ratio."""
    if not have("zstandard"):
        return {"skipped": True, "error": "zstandard not installed"}
    import zstandard as zstd

    data = _payload(4 * MB)
    comp = zstd.ZstdCompressor(level=3)
    decomp = zstd.ZstdDecompressor()
    blob = comp.compress(data)
    if decomp.decompress(blob) != data:
        raise ValidationError("zstd: round trip mismatch — possible fault")

    r = _rate(lambda: comp.compress(data), len(data), seconds, repeats)
    return {"unit": "MB/s", "rate": r["rate"], "cv": r["cv"],
            "ratio": round(len(data) / len(blob), 2),
            "level": 3, "validated": True}


def bench_lz4(seconds: float = 0.6, repeats: int = 2) -> dict:
    """LZ4 compression throughput — speed over ratio."""
    if not have("lz4"):
        return {"skipped": True, "error": "lz4 not installed"}
    import lz4.frame

    data = _payload(4 * MB)
    blob = lz4.frame.compress(data)
    if lz4.frame.decompress(blob) != data:
        raise ValidationError("lz4: round trip mismatch — possible fault")

    r = _rate(lambda: lz4.frame.compress(data), len(data), seconds, repeats)
    return {"unit": "MB/s", "rate": r["rate"], "cv": r["cv"],
            "ratio": round(len(data) / len(blob), 2), "validated": True}


def bench_blake3(seconds: float = 0.6, repeats: int = 2) -> dict:
    """BLAKE3 hashing — a modern, heavily parallel hash."""
    if not have("blake3"):
        return {"skipped": True, "error": "blake3 not installed"}
    import blake3 as b3

    data = _payload(4 * MB)
    expected = b3.blake3(data).hexdigest()

    def chunk():
        if b3.blake3(data).hexdigest() != expected:
            raise ValidationError("blake3: digest mismatch — possible fault")

    r = _rate(chunk, len(data), seconds, repeats)
    return {"unit": "MB/s", "rate": r["rate"], "cv": r["cv"],
            "validated": True}


def run(seconds: float = 0.6, repeats: int = 2) -> dict:
    """Every available crypto and compression benchmark."""
    avail = available()
    if not any(avail.values()):
        return {"available": False,
                "note": "no crypto/compression packages — run "
                        "'python3 install.py --tier crypto'"}
    out: dict = {"available": True, "packages": avail}
    for name, fn in (("aes", bench_aes), ("zstd", bench_zstd),
                     ("lz4", bench_lz4), ("blake3", bench_blake3)):
        try:
            result = fn(seconds, repeats)
        except ValidationError as e:
            result = {"error": str(e), "validation_failed": True}
        except Exception as e:
            result = {"error": f"{type(e).__name__}: {e}"}
        if not result.get("skipped"):
            out[name] = result
    return out


def extract_rates(payload: dict | None) -> dict:
    if not payload or not payload.get("available"):
        return {}
    rates = {}
    for key, score_key in (("aes", "aes"), ("zstd", "zstd"),
                           ("lz4", "lz4"), ("blake3", "blake3")):
        entry = payload.get(key) or {}
        if isinstance(entry.get("rate"), (int, float)) and entry["rate"] > 0:
            rates[score_key] = float(entry["rate"])
    return rates
