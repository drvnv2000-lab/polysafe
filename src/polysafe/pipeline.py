"""One command: load -> label -> features -> exposure -> models -> sensitivity -> exploratory
-> secondary -> figures -> integrity report -> Research Lab page (HANDOFF non-negotiable 3).

Every step rewrites its committed results; on unchanged code and data a rerun leaves
``git diff results/`` empty, which is the reproducibility check. The power and coverage
simulations (section 9, D2.1a) are slow and run only with ``--simulations``. The mapping
draft and review packet (``medmap``, ``review``) are never rebuilt here: the review is the
PI's adjudicated artifact.

Usage:
    python -m polysafe.pipeline [--workers 6] [--simulations]
"""

from __future__ import annotations

import argparse
import os
import time

from polysafe import (
    ascertain,
    coverage,
    exploratory,
    features,
    figures,
    integrity,
    lab,
    labels,
    models,
    power,
    secondary,
    sensitivity,
)


def steps(workers: int, simulations: bool) -> list[tuple[str, object]]:
    w = ["--workers", str(workers)]
    out = [("labels", lambda: labels.main([])),
           ("features", lambda: features.main([])),
           ("ascertain", lambda: ascertain.main([]))]
    if simulations:
        out += [("power", lambda: power.main([])), ("coverage", lambda: coverage.main([]))]
    out += [("models", lambda: models.main(w)),
            ("sensitivity", lambda: sensitivity.main(w)),
            ("exploratory", lambda: exploratory.main(w)),
            ("secondary", lambda: secondary.main([])),
            ("figures", lambda: figures.main([])),
            ("integrity", integrity.main),
            ("lab", lambda: (lab.build_lab(), 0)[1])]
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the whole PolySafe pipeline.")
    p.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 2)))
    p.add_argument("--simulations", action="store_true")
    args = p.parse_args(argv)
    for name, step in steps(args.workers, args.simulations):
        t0 = time.time()
        print(f"== {name}", flush=True)
        code = step()
        if code:
            print(f"{name} exited {code}; stopping")
            return int(code)
        print(f"== {name} done in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
