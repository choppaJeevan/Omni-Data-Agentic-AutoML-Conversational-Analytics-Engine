# ADA — Agentic Data Analysis: Project Overview

![ADA Architecture Flowchart](ADA%20flowchart.png)

## What is ADA?

ADA (Agentic Data Analysis) is a **fully local, autonomous AI data scientist**. You hand it a raw CSV file, and it independently profiles the data, infers the prediction target, cleans and encodes the features, decides which family of machine learning models fits the data's geometry, runs a hyperparameter-tuned model tournament, audits its own winner for overfitting and data leakage, writes its own exploratory analysis code, and compiles a professional executive summary report — all without a single API call leaving your machine.

Every "thinking" step is powered by small open-source LLMs running locally through **Ollama**: Llama 3 for text reasoning and LLaVA for visual judgment. Every "doing" step is powered by deterministic, battle-tested Python tooling (pandas, scikit-learn, statsmodels, XGBoost, CatBoost, LightGBM).

## Design Philosophy

The architecture follows one core rule: **let the LLM decide, let deterministic code execute.**

- LLMs are never trusted to do math, transform data, or train models. They only make *routing decisions* (which target column? linear or tree geometry?) and *interpretations* (is this model healthy? what does this report say?).
- Deterministic functions (the "Engine Room") do all the heavy lifting — preprocessing, leakage detection, training, validation — so results are reproducible and auditable.
- A hard-coded router (not an LLM) handles the regression vs. classification branch, eliminating hallucinated control flow where statistics already dictate the answer.

## The Pipeline, Node by Node

The flowchart above traces the full journey of a CSV through the system:

### 1. Data Scout
Extracts a compact JSON metadata summary of the CSV — shape, dtypes, missing values, cardinality, sample values. This keeps a 100MB dataset down to a few KB of context so a local 8B-parameter LLM can reason about it without exhausting VRAM.

### 2. Target Inference
Llama 3 reads the metadata summary and infers which column is the machine learning target — no human labeling required.

### 3. Global Preprocessor
A deterministic cleanup pipeline that splits train/test *first* (to prevent leakage), then:
- Drops ID-like columns (numeric or categorical with ≥40% unique values)
- Imputes missing values (median / mode, fitted on train only)
- Adaptive encoding: one-hot for low cardinality, frequency encoding for medium cardinality
- **Three-layer leakage guard**: drops categorical columns that perfectly partition the target, features with |correlation| ≥ 0.95, and groups of binary flags that collectively reconstruct the target (F1 ≥ 0.90 on flags alone)
- Detects and removes duplicate columns

### 4. Dynamic Task Router
A deterministic switch: regression goes one way, classification goes the other.

### 5a. Regression Branch — The Visual Judge
Fits a baseline OLS model with statsmodels and saves a residual plot. Then **LLaVA, a vision LLM, literally looks at the plot** — if it sees a random cloud, the data is linear; if it sees curvature or a heteroscedasticity funnel, it routes to tree-based models. The agent uses its *eyes* the way a human data scientist would.

### 5b. Classification Branch — The Statistical Judge
Benchmarks Logistic Regression vs. Random Forest with cross-validated Macro F1 (imbalance-safe). Llama 3 reads the F1 delta and rules whether the decision boundary is linear or non-linear.

### 6. Conditional Optimizer
If a LINEAR model family was chosen, the data gets "surgery": IQR outlier removal and RobustScaler scaling (fitted on train only). Tree models skip this entirely — they're scale- and outlier-invariant, so the work would be wasted.

### 7. Model Tournament
The chosen family competes in a hyperparameter-tuned tournament via `RandomizedSearchCV`:
- **Linear suite**: LinearRegression / Ridge / Lasso, or LogisticRegression / LinearSVC
- **Tree suite**: RandomForest, GradientBoosting, KNN, XGBoost, CatBoost, LightGBM

The winner is selected on R² (regression) or Macro F1 (classification), and predictions are exported to CSV.

### 8. Critic / Validator Agent
The system audits its own work. It computes train-vs-test score gaps and flags `OVERFITTING_DETECTED`, `UNDERFITTING_DETECTED`, or `DATA_LEAKAGE_SUSPECTED`. It generates a full diagnostic plot suite — confusion matrix, ROC-AUC curves, precision-recall curves, learning curves — then Llama 3 writes a natural-language health verdict.

### 9. Sandbox Code Generator (Self-Correcting)
Llama 3 writes its own exploratory Python analysis script, which runs in an isolated sandbox namespace. If the code crashes, the full traceback is fed back to the LLM, which fixes its own bug — up to 3 attempts. `plt.show()` is intercepted so generated code can never freeze the terminal.

### 10. Insight Synthesizer & Reporter
All state artifacts — routing decisions, metrics, critic flags, execution history — are compiled by Llama 3 into a polished Markdown executive summary report saved to `reports/`.

### 11. Interactive Q&A Workspace
After the pipeline completes, you can chat with your data. The agent answers questions grounded in the actual pipeline ledger and a data snapshot. If you ask for a plot or computation, it **writes and executes the code live** in the sandbox, with the same self-correction loop.

## State Management

A single Pydantic model (`AnalystGraphState`) is the system's central nervous system. Every node reads from it and writes typed updates back, producing a full execution audit trail. Completed pipelines are cached to disk (keyed by dataset hash) so re-runs can fast-forward straight to the Q&A workspace.

## Proven Results

| Dataset | Task | Winning Model | Test Score | Health Verdict |
|---|---|---|---|---|
| Telco Customer Churn | Classification | CatBoost | 0.797 accuracy | HEALTHY (gap 0.037) |
| Titanic | Classification | Tree family | 0.835 accuracy | HEALTHY |

## File Map

| File | Role |
|---|---|
| `graph_backbone.py` | Orchestrator: all graph nodes, routing, main pipeline loop, Q&A workspace |
| `engine_room.py` | Deterministic tools: metadata extractor, preprocessor, OLS baseline, optimizer, benchmarker, critic engine |
| `state_schema.py` | Pydantic central graph state |
| `sandbox_executor.py` | Isolated execution environment for AI-generated code |
| `reports/` | Generated executive reports, prediction CSVs, AI-written scripts, pipeline caches |
| `plots/` | Residual plots and diagnostic visuals |
