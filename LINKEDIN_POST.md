I built an AI Data Scientist that runs 100% on my own laptop. No OpenAI. No cloud. No API keys.

I call it ADA — Agentic Data Analysis.

You give it one thing: a raw CSV file.

It does everything else on its own:

🔹 Scans the dataset and infers the prediction target: Llama 3 reads the schema and picks the target column — no human labeling required.
🔹 Cleans, encodes, and splits the data: Features a 3-layer data-leakage guard that catches proxy variables most pipelines silently train on.
🔹 Determines the problem type: Automatically detects if it's a regression or classification task.
🔹 Visual Diagnostics (My favorite part!): If it's regression, it fits a linear model and plots the residuals. Then, a vision model (LLaVA) literally LOOKS at the plot. Random cloud? It stays with linear models. Curved or funnel-shaped? It routes to tree-based models. The agent uses its "eyes" exactly how a human data scientist would.
🔹 Intelligent Routing: If it's classification, it evaluates logistic regression against tree-based models, sends the metrics back to Llama 3, and lets the LLM deduce if the decision boundary is linear or non-linear.
🔹 Runs a Hyperparameter-Tuned Tournament: XGBoost vs. CatBoost vs. LightGBM vs. Random Forest vs. Gradient Boosting vs. KNN. The best model wins.
🔹 Self-Auditing: A Critic Agent evaluates the train/test gap and raises flags like OVERFITTING_DETECTED, UNDERFITTING_DETECTED, or DATA_LEAKAGE_SUSPECTED with full diagnostics (ROC, PR curves, learning curves, confusion matrix).
🔹 Self-Healing Code: It writes its own Python analysis code, runs it in a sandbox, reads its own error tracebacks, and fixes its own bugs.
🔹 Interactive Reporting: Finally, it compiles a professional executive summary and opens a chat where you can ask your data anything — writing and executing plotting code live.

Real result: On the Telco Customer Churn dataset, ADA autonomously detected a non-linear boundary, selected CatBoost, hit 79.7% test accuracy, and certified itself HEALTHY with a 0.037 train/test gap.

The core design rule that made this work:
LLMs make decisions. Deterministic code does the math. 
The models never touch the data transformations — they only choose routes and interpret results. That’s what keeps it reproducible.

Tech stack: Python, Ollama (Llama 3 + LLaVA), Pydantic state graph, scikit-learn, statsmodels, XGBoost, CatBoost, LightGBM.

Everything runs locally. Your data never leaves your machine. 

The full architecture is in the flowchart below. Happy to share details if you're building agentic pipelines too! 👇

#MachineLearning #AIAgents #DataScience #LocalLLM #Ollama #Python #LLM #AgenticAI #OpenSource #MLOps