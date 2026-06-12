# ADA — Agentic Data Analysis

> A fully local, autonomous AI data scientist. Drop in a CSV — get back a trained, validated, self-audited ML model and an executive report. Zero cloud. Zero API keys. 100% private.

![ADA Architecture Flowchart](ADA%20flowchart.png)

## Features

- **Autonomous end-to-end pipeline** — profiling, target inference, preprocessing, model selection, tuning, validation, and reporting with no human intervention
- **AI target inference** — a local Llama 3 model reads the dataset schema and picks the target column itself
- **Vision-based model selection** — a LLaVA vision model *looks at* the OLS residual plot and decides between linear and tree-based architectures, just like a human data scientist would
- **Three-layer data leakage guard** — catches target-partitioning categories, high-correlation proxies, and groups of binary flags that collectively reconstruct the label
- **Model tournament** — RandomizedSearchCV-tuned competition between up to 6 algorithms (RandomForest, GradientBoosting, KNN, XGBoost, CatBoost, LightGBM) or linear suites (Ridge, Lasso, LogisticRegression, LinearSVC)
- **Self-auditing Critic Agent** — flags overfitting, underfitting, and suspected leakage; generates confusion matrix, ROC-AUC, precision-recall, and learning curve diagnostics
- **Self-correcting code generation** — the LLM writes its own analysis scripts, runs them in a sandbox, reads its own tracebacks, and fixes its own bugs (up to 3 retries)
- **Interactive data Q&A** — chat with your dataset after the pipeline finishes; the agent writes and executes plotting/analysis code on demand
- **Pipeline caching** — completed runs are cached so you can jump straight back into Q&A without retraining
- **100% local & private** — all reasoning runs through Ollama on your own machine

## Architecture

The system is split into two layers (see `OVERVIEW.md` for a deep dive):

| Layer | What it does | Powered by |
|---|---|---|
| **Brain** (decisions) | Target inference, linear-vs-tree judgment, health verdicts, report writing | Llama 3 + LLaVA via Ollama |
| **Engine Room** (execution) | Preprocessing, leakage guards, training, tuning, validation, plotting | pandas, scikit-learn, statsmodels, XGBoost, CatBoost, LightGBM |

```
CSV → Data Scout → Target Inference → Global Preprocessor → Router
        ├─ Regression  → OLS Baseline → Visual Judge (LLaVA reads the residual plot)
        └─ Classification → CV Benchmarker → Statistical Judge (Llama 3 reads the F1 delta)
                  ↓
   Conditional Optimizer (outlier surgery + robust scaling, LINEAR only)
                  ↓
   Model Tournament (RandomizedSearchCV) → Critic/Validator Agent
                  ↓
   Sandbox Code Generator (self-correcting) → Executive Report → Interactive Q&A
```

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally
- Local models pulled:

```bash
ollama pull llama3   # text reasoning
ollama pull llava    # vision judgment (regression branch)
```

- Python dependencies:

```bash
pip install pandas numpy scikit-learn statsmodels matplotlib seaborn pydantic requests xgboost catboost lightgbm
```

## Quick Start

1. Start the Ollama engine:

```bash
ollama serve
```

2. Run the agent:

```bash
python graph_backbone.py
```

3. When prompted, paste the path to your CSV file (drag-and-drop into the terminal works too):

```
Enter the local path to your dataset (.csv): Telco-Customer-Churn.csv
```

4. Watch the pipeline run — then ask questions:

```
Ask a question about your data: which features drive churn the most?
Ask a question about your data: plot the distribution of monthly charges
```

Type `exit` to quit. Re-running on the same dataset offers to load the cached pipeline and skip training.

## Outputs

| Artifact | Location |
|---|---|
| Executive summary report (Markdown) | `reports/<dataset>_final_data_report.md` |
| Predictions (actual vs. predicted) | `reports/<dataset>_predictions_output.csv` |
| AI-generated analysis scripts | `reports/<dataset>_sandbox_script.py`, `reports/<dataset>_qa_script.py` |
| Diagnostic plots (confusion matrix, ROC, PR, learning curve, residuals) | `plots/` |
| Pipeline cache for fast resume | `reports/<dataset>_pipeline_cache_<hash>.json` |

## Project Structure

```
agentic_data_analysis/
├── graph_backbone.py     # Orchestrator: graph nodes, routing, main loop, Q&A
├── engine_room.py        # Deterministic tools (Functions 1–6)
├── state_schema.py       # Pydantic central graph state
├── sandbox_executor.py   # Isolated runtime for AI-generated code
├── ADA flowchart.png     # Full architecture flowchart
├── reports/              # Generated reports, predictions, scripts, caches
└── plots/                # Diagnostic visuals
```

## Example Result

On the Telco Customer Churn dataset, ADA autonomously:
- Inferred `Churn` as the target
- Detected a non-linear decision boundary and routed to the tree family
- Crowned **CatBoost** the tournament winner (79.7% test accuracy)
- Verified model health: train/test gap of 0.037 — no overfitting, no leakage

## License

This project is for educational and portfolio purposes.
