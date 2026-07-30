# Requirements Traceability

| ID | Requirement | Implementation | Verification |
|---|---|---|---|
| R01 | RQ1 is successful when a working Dashboard is delivered and displayed | Page 1 completion statement; eight-section `app.py` | AppTest Page 1 and navigation checks |
| R02 | Use traceable TSLA, AAPL, and NVDA daily data | `data/raw/`; Page 1 source display; research snapshots | Data tests and SYS-05 |
| R03 | Explain why each RQ2 feature is selected | Page 3 six-row rationale; `src/literature.py` | AppTest unique six-row table |
| R04 | Support every method area with literature | Pages 1, 3, 4, 5, 6, 8; literature document | Literature coverage test |
| R05 | Compare four feature families fairly | RQ2 common regularised LR and temporal protocol | RQ2 validator |
| R06 | Use ML models including LSTM | LR, RF, XGBoost, Compact LSTM on Page 4 | RQ3 files and AppTest |
| R07 | Compare AAPL and NVDA with TSLA | Page 5 common-protocol, independently fitted replication | Complete 3 × 4 table |
| R08 | Close RQ2 → RQ3 → RQ4 without return-driven reselection | Contract and reviewed result flow | RQ3/RQ4 validators |
| R09 | Present formal results without presentation-time retraining | Pages 3–6 read packaged outputs | No formal-page training-button test |
| R10 | Keep optional CSV analysis | Page 7 bundled/uploaded data and explicit run button | Sandbox AppTest |
| R11 | Include costs and transparent references in RQ4 | Five sleeves, 15 bps scenario, B&H, EMA, SMA | RQ4 protocol and validator |
| R12 | Keep package internally verifiable | 84 analytical checks, system checks, root manifest | Validation scripts |

The canonical research wording and evidence paths are in
`research_contract.json`. The detailed source mapping is in
`docs/literature_and_design_rationale.md`.
