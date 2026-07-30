# Changelog

## v8.1-deploy — GitHub and Streamlit Community Cloud edition

- Added a pinned, lightweight Python 3.12 cloud dependency profile.
- Kept TensorFlow in the complete local profile while preserving all reviewed
  Compact LSTM evidence in the cloud Dashboard.
- Made Page 7 automatically hide only live LSTM training when TensorFlow is
  unavailable and explain that Pages 4–6 remain complete.
- Disabled cloud-mode hyperparameter searches and capped tree estimators while
  retaining the three lightweight live Sandbox models.
- Added GitHub Actions checks for release integrity, Streamlit page behaviour,
  and the `/_stcore/health` endpoint.
- Added current GitHub, Community Cloud, acceptance, wake-up, local-backup, and
  rollback instructions.
- Set private repository and examiner-email access as the default until the
  bundled Nasdaq snapshot rights are cleared for public deployment.
- Added secret-file exclusions and deliberately omitted the unsupported
  `runtime.txt` convention.

## v8.1 — Final examiner release

- Synchronised the RQ2 README and formal methodology/results document with the
  six-feature literature rationale shown in the Dashboard.
- Added Petrozziello et al. (2022) as direct recent precedent for a 20-day LSTM
  look-back while retaining the project's fair-comparison claim boundary.
- Replaced stale launch text and absolute validation paths with current,
  portable release wording.
- Made RQ3 and RQ4 validators read-only by default; intentional regeneration
  now requires `--write-artifacts`.
- Gave Development and Audit bars distinct colours, used a fixed 60–70% scale
  for the cross-asset Balanced-Accuracy heatmap, and made RQ4 scope metrics
  follow the active filters.

## v8.0.5 — Current literature and browser-stable presentation

- Replaced every pre-2020 Dashboard reference with literature published from
  2020 onward and strengthened the feature-by-feature RQ2 traceability.
- Removed all explicit download widgets and disabled built-in DataFrame export.
- Stabilised RQ4 widget identity across filter changes and documented
  extension/translation-related React `removeChild` recovery.

## v8.0.4 — Examiner-ready literature and lean release

- Made Page 1 explicitly answer RQ1 through delivery of the working Dashboard.
- Added section-specific literature support across the RQ1–RQ4 workflow.
- Added a direct six-row RQ2 feature rationale with definitions, inclusion
  reasons, sources, and honest parameter boundaries.
- Added complete APA-style literature traceability on Page 8 and in
  `docs/literature_and_design_rationale.md`.
- Kept the later-period RQ4 result available while reducing its presentation to
  one restrained limitations caption.
- Removed unreachable interface code and stale operational documentation.
- Removed duplicated processed data and unused historical supporting assets
  from the delivery package.
- Preserved all reviewed RQ2, RQ3, RQ4 protocols, predictions, ledgers,
  validators, and formal result files.
- No formal LR, RF, XGBoost, or LSTM model was retrained for this release.
