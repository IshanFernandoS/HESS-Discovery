"""Manuscript-consistent GMM basin evidence and Eq. (1)-(5) selection.

This file is the public GMM entry point for the revised manuscript. It delegates
to ``validations.equation_1_5_selection`` so that the repository has one
canonical implementation of the corrected workflow:

1. fit/calibrate the GMM in descriptor space;
2. compute descriptor evidence and the unweighted basin score B(c);
3. retain the top rho fraction within each cation order;
4. rank retained candidates by the frozen manuscript GNN EFA prediction.

The retired z-scored posterior/coverage/cumulative-posterior composite score is
not used here.
"""

from __future__ import annotations

import sys
from pathlib import Path


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validations.equation_1_5_selection import main


if __name__ == "__main__":
    main()
