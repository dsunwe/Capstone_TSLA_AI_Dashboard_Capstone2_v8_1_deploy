# Testing and Validation

Validation is split into four layers.

## 1. Formal analytical checks

| Module | Checks |
|---|---:|
| RQ2 | 22/22 |
| RQ3 | 24/24 |
| RQ4 | 38/38 |
| **Combined** | **84/84** |

These validators check protocols, data boundaries, feature order, prediction
rows, signal timing, sleeve trades, metrics, and stored-file relationships.

## 2. Streamlit interaction checks

`tests/test_app_smoke.py` verifies:

- exactly eight navigable sections;
- Page 1 visibly answers RQ1;
- Page 3 contains the six-row feature-and-literature rationale;
- formal RQ3/RQ4 pages do not expose training buttons;
- TSLA/AAPL/NVDA and all four ML models are present;
- Development/audit and asset filters update saved rows; and
- Page 7 training is explicit and isolated.

## 3. Integrated system checks

`scripts/validate_integrated_release.py` checks the research contract, shared
target and purge, three-asset data, one-SE selection, complete RQ4 grid,
current navigation, literature coverage, and required launch/reproduction
entry points.

## 4. Release integrity

`scripts/build_root_release_manifest.py` records every packaged file, byte
size, and SHA-256 digest. `scripts/verify_root_release_manifest.py` checks the
delivered set against that manifest.

## Commands

```bash
python -m pytest
python research/rq2/validate_rq2.py
python research/rq3/validate_rq3.py
python research/rq4/validate_rq4.py
python scripts/validate_integrated_release.py
python scripts/verify_root_release_manifest.py
```

The RQ3 and RQ4 commands above are read-only: they recompute and print every
check but do not overwrite the packaged validation evidence. Use
`--write-artifacts` only after an intentional formal recomputation, then
rebuild the root release manifest.

Passing checks establish technical consistency and reproducibility. They are
not a guarantee of future market performance.
