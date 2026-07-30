# GitHub and Streamlit Community Cloud Deployment

This is the recommended examiner demonstration path for the v8.1 Capstone
Dashboard. The formal pages load reviewed bundled results and do not retrain
LR, RF, XGBoost, or LSTM during navigation.

## 0. Access and data-rights default

Use a **private GitHub repository and private Community Cloud app**. The
bundled `data/context/source_manifest.csv` explicitly labels the Nasdaq
TSLA/AAPL/NVDA/QQQ snapshots as private academic evaluation material and says
to review Nasdaq terms before public deployment. Do not make the repository or
app public until the right to redistribute and publicly display both source
and derived data has been confirmed.

Community Cloud supports inviting the examiner by email. The examiner does not
need repository access or a GitHub account, but must authenticate through
Google or a single-use email link. Community Cloud currently permits one
private app at a time, and an invited viewer can invite additional viewers, so
use examiner-controlled email addresses.

Official references:

- [Connect a private GitHub repository](https://docs.streamlit.io/deploy/streamlit-community-cloud/get-started/connect-your-github-account)
- [Share a private app](https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app)
- [Trust and security](https://docs.streamlit.io/deploy/streamlit-community-cloud/get-started/trust-and-security)

## 1. Deployment profile

| File | Purpose |
|---|---|
| `requirements.txt` | Lightweight Community Cloud runtime: Streamlit, data, charting, LR/RF/XGBoost |
| `requirements-local.txt` | Complete local runtime, including TensorFlow and pytest |
| `.streamlit/config.toml` | Shared theme, headless server mode, and disabled table export |
| `.github/workflows/deploy-smoke.yml` | Ubuntu/Python 3.12 integrity, page, and health checks |
| `app.py` | Streamlit entry point |

TensorFlow is intentionally absent from the cloud runtime. This avoids making
live neural-network training a dependency of a free presentation instance.
Pages 4–6 still show the complete reviewed Compact LSTM evidence. Page 7
automatically hides the live LSTM option and explains the boundary. Public CSV
uploads are capped at 10 MB and 10,000 rows; data and model caches have bounded
entry counts and one-hour expiry. Live hyperparameter search is disabled and
the tree-estimator control is capped at 240 in this profile.

## 2. Upload to GitHub

Create an empty **private** GitHub repository, then run these commands from
this folder:

```bash
git init
git add .
git commit -m "Deploy Capstone II v8.1"
git branch -M main
git remote add origin https://github.com/YOUR-ACCOUNT/YOUR-REPOSITORY.git
git push -u origin main
```

GitHub Desktop is also suitable. Upload the folder contents, not the ZIP and
not an extra parent directory. The repository root must directly contain
`app.py`, `requirements.txt`, `.streamlit/`, `data/`, `research/`, and `src/`.

## 3. Create the Community Cloud app

1. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/).
2. Select **Create app**.
3. Choose the GitHub repository and `main` branch.
4. Set **Main file path** to `app.py`.
5. Open **Advanced settings** and explicitly select **Python 3.12**.
6. Leave Secrets empty; this bundled demonstration needs no credentials.
7. Deploy.
8. In **App settings → Sharing**, keep **Only specific people can view this
   app** and add the examiner's email.

Python is selected in the deployment interface. Do not add `runtime.txt`.
Community Cloud discovers the root `requirements.txt`; adding another
higher-priority environment file such as `uv.lock`, `Pipfile`, or
`environment.yml` may cause a different environment to be used.

Official references:

- [Deploy an app](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [App dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)
- [File organisation](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization)
- [Secrets management](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management)
- [Resource management and sleep behaviour](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app)

## 4. First-deployment acceptance check

Verify the following after the build finishes:

1. the app opens without a dependency or file-not-found error;
2. Page 1 displays the RQ1 completion statement and TSLA data range;
3. Page 3 shows four feature families and the six-feature rationale;
4. Page 4 shows all four formal models, including Compact LSTM;
5. Page 5 switches among TSLA, AAPL, and NVDA evidence;
6. Page 6 filters period, asset, and model correctly;
7. Page 7 offers Logistic Regression, Random Forest, and XGBoost and shows the
   live-LSTM capability notice; and
8. Page 8 shows RQ2 `22/22`, RQ3 `24/24`, RQ4 `38/38`.

The cloud app can sleep after a period without traffic. Open and check it
before the presentation rather than relying on the first examiner visit to
wake it.

## 5. Automatic GitHub checks

The included workflow runs on pushes and pull requests. It installs the same
cloud dependencies on Ubuntu with Python 3.12, verifies the release manifest,
runs deployment-focused tests, launches Streamlit, and checks
`/_stcore/health`.

Because `release_manifest.json` covers the complete package, an intentional
file change requires rebuilding it before committing:

```bash
python scripts/build_root_release_manifest.py
python scripts/verify_root_release_manifest.py
```

## 6. Local full environment and backup

Keep the original local v8.1 or use the complete profile in this package:

### macOS / Linux

```bash
chmod +x setup_macos_linux.sh run_dashboard.sh
./setup_macos_linux.sh
./run_dashboard.sh
```

### Windows

```bat
setup_windows.bat
run_dashboard.bat
```

Manual macOS/Linux setup:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-local.txt
.venv/bin/python -m streamlit run app.py --server.port 8501
```

The local profile includes TensorFlow and restores live LSTM Sandbox training.
It is the recommended offline presentation backup.

## 7. Browser stability

The Dashboard does not inject custom JavaScript or manipulate the browser DOM.
If React reports a `removeChild` `NotFoundError`, reload once in a Guest or
private window with page translation, grammar tools, and other DOM-modifying
extensions disabled.

After changing `.streamlit/config.toml` or browser settings:

1. restart Streamlit or reboot the cloud app;
2. hard-refresh the page; and
3. test once with extensions disabled.

All explicit Dashboard download controls and Streamlit's built-in DataFrame
export menu remain disabled.

## 8. Rollback

Create a Git tag after the first accepted deployment:

```bash
git tag v8.1-deploy
git push origin v8.1-deploy
```

If a later commit breaks the cloud app, redeploy a known-good branch or revert
the offending commit. Do not rerun formal RQ2–RQ4 experiment scripts inside
the deployed repository; those scripts can intentionally regenerate evidence.
