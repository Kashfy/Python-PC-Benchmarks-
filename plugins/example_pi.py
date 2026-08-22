"""Example plugin: estimate pi by the Leibniz series.

Copy this file, change the maths, and it appears in the next run — timed,
scored, printed, and written to the CSV with no other changes.

The three required attributes are NAME, UNIT and BASELINE, plus a
``run(seconds, repeats)`` returning a dict containing ``rate``.
"""

import time

NAME = "Pi (Leibniz series)"
UNIT = "terms/s"
BASELINE = 5_000_000.0          # rate corresponding to a score of 100

TERMS_PER_CHUNK = 200_000


def _chunk() -> float:
    total, sign = 0.0, 1.0
    for k in range(TERMS_PER_CHUNK):
        total += sign / (2 * k + 1)
        sign = -sign
    return total * 4.0


def run(seconds: float, repeats: int) -> dict:
    # Correctness check: the series must converge near pi. A wrong answer here
    # means the same thing as in any built-in benchmark — suspect the hardware.
    estimate = _chunk()
    if abs(estimate - 3.14159265) > 0.001:
        raise ValueError(f"pi estimate {estimate} is not close to pi")

    rates = []
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        n = 0
        while time.perf_counter() - start < seconds:
            _chunk()
            n += 1
        elapsed = time.perf_counter() - start
        rates.append(n * TERMS_PER_CHUNK / elapsed)

    rates.sort()
    return {
        "rate": rates[len(rates) // 2],
        "estimate": round(estimate, 8),
        "terms_per_chunk": TERMS_PER_CHUNK,
    }
