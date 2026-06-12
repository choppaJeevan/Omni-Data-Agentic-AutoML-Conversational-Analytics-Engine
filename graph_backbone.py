import json 
import base64 
import os
import requests
import warnings
import logging

# Mute all noisy Machine Learning library warnings from cluttering the terminal
warnings.filterwarnings("ignore")
logging.getLogger("lightgbm").setLevel(logging.ERROR)
os.environ['PYTHONWARNINGS'] = 'ignore'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from state_schema import AnalystGraphState
import engine_room as er 
from sandbox_executor import execute_sandboxed_code
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, f1_score, accuracy_score
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

OLLAMA_URL = "http://localhost:11434/api/generate"

# Component 2
def query_local_text_llm(prompt: str, model_name: str = "llama3") -> str:
    """
    Sends a structural text generation request to your local ollama instance.
    """

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 1024} # selecting the token for highly deterministic.
    }
    try:
        response = requests.post(OLLAMA_URL, json = payload, timeout = 300)
        return response.json().get("response", "").strip()
    except Exception as e:
        return f"Error_connecting_to_ollama: {str(e)}"


def query_local_vision_llm(prompt: str, image_path: str, model_name: str = "llava") -> str:
    """
    Encodes a local PNG file to Base64 and queries your local multi-modal model.
    """
    if not os.path.exists(image_path):
        return f"Error: Image file not found at {image_path}"

    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    payload = {
        "model": model_name,
        "prompt": prompt,
        "images": [encoded_image],
        "stream": False,
        "options": {"temperature": 0.2}
    }

    try:
        response = requests.post(OLLAMA_URL, json = payload, timeout = 90)
        return response.json().get("response", "").strip()
    except Exception as e:
        return f"Error_connecting_to_OLLAMA_vision: {str(e)}"

# Graph Nodes
def data_scout_node(state: AnalystGraphState) -> dict:
    """
    Executes Function 1 to extract technical dataset metadata structural context.
    """
    print("\n [Node: Data Scout] scanning CSV structure via Function 1")
    meta_json = er.extract_metadata(state.csv_path)

    #update the tracking history and store metadata summary dict
    return {
        "metadata_summary": json.loads(meta_json),
        "execution_history": state.execution_history + ["data_scout"]
    }

def target_inference_node(state: AnalystGraphState) -> dict:
    """
    Ask the local text LLM to analyze the metadata summary and pick the best target variable. 
    """
    print("[Node: Target Inference] AI text model is extrcating target objective semantic signatures")

    prompt = f"""
    You are an expert Data Sciencist Supervisor. Analyze this dataset metadata summary profile schema:
    {json.dumps(state.metadata_summary, indent = 2)}

    Identify which column name is most logically intended to be the machine learning target variable (e.g., Target_value, label, price, status).
    Respond strictly with just the raw column name, nothing else. Do not provide a sentence, punctuation, or explanations.
    """
    inferred_target = query_local_text_llm(prompt)
    print(f"AI inferred Target Variable: {inferred_target}")

    return {
        "target_column": inferred_target,
        "execution_history": state.execution_history + ["target_inference"]
    }

def global_preprocessor_node(state: AnalystGraphState) -> dict:
    """
    Executes Function 2 cleanup using the dynamically derived target column.
    """
    print(" [Node: Global Preprocessor] Stripping missing blocks & applying dynamic encodings.")

    # Call our structural Part 1 global preprocessing engine (train-test split is done here)
    _, _, _, _, problem_type = er.global_preprocess(state.csv_path, target_column=state.target_column)
    
    # We store the problem_type to control our backbone gateway router
    return {
        "problem_type": problem_type,
        "execution_history": state.execution_history + ["global_preprocessor"]
    }

def regression_visual_judge_node(state: AnalystGraphState) -> dict:
    """
    Part 4: The Visual Judge.
    Uses a local multi-modal Vision model (LLaVA) to look at the residual plot 
    and identify geometric violations of OLS assumptions.
    """
    print("\n [Node: Visual Judge] Analyzing residual plot structure using LLaVA...")
    
    # We first run our Function 3 tool to ensure the baseline plot exists
    # For our mock testing script, we pull the path directly from our state configuration
    plot_path = state.residual_plot_path or "residual_plot.png"
    
    prompt = """
    You are a Senior Statistical Vision System. Analyze this residual plot generated from a baseline OLS linear regression model.
    Look closely at how the scatter points spread from left to right along the zero center-line:
    
    1. If the points form a random, uniform cloud, output strictly: CHOICE: LINEAR
    2. If the points form a clear curved shape, a parabola, or a distinct 'U' pattern, output strictly: CHOICE: TREE
    3. If the points form an expanding funnel/megaphone shape where variance grows wider, output strictly: CHOICE: TREE
    
    Examine carefully, then output your final choice on a single line starting with 'CHOICE: ' followed by either LINEAR or TREE.
    """
    
    vision_decision = query_local_vision_llm(prompt, image_path=plot_path)
    print(f" Visual Judge Analysis Output:\n{vision_decision}")
    
    # Simple keyword routing parser
    chosen_family = "TREE" if "TREE" in vision_decision.upper() else "LINEAR"
    print(f" AI Architectural Selection: {chosen_family}")
    
    return {
        "chosen_model_family": chosen_family,
        "execution_history": state.execution_history + ["regression_visual_judge"]
    }

def classification_statistical_judge_node(state: AnalystGraphState) -> dict:
    """
    Part 4: The Statistical Judge.
    Parses the Macro F1 benchmark delta to determine if structural boundaries are non-linear.
    """
    print("\n [Node: Statistical Judge] Analyzing Macro F1 benchmarks using Llama 3...")
    
    benchmarks = state.classification_benchmarks
    
    prompt = f"""
    You are a Staff Machine Learning Critic. Review these cross-estimator performance benchmarks:
    {json.dumps(benchmarks, indent=2)}
    
    Evaluate the 'delta' parameter (Tree F1 score minus Linear Baseline F1 score). 
    - If the tree model outperforms the linear baseline even slightly (delta > 0.02), the data space is likely non-linear.
    - If the delta is tiny or negative, the linear relationship holds perfectly.
    
    Output exactly one of these options on a single line:
    CHOICE: TREE (if delta is greater than 0.02)
    CHOICE: LINEAR (if delta is less than or equal to 0.02)
    """
    
    text_decision = query_local_text_llm(prompt)
    print(f" Statistical Judge Analysis Output: {text_decision}")
    
    chosen_family = "TREE" if "TREE" in text_decision.upper() else "LINEAR"
    print(f" AI Architectural Selection: {chosen_family}")
    
    return {
        "chosen_model_family": chosen_family,
        "execution_history": state.execution_history + ["classification_statistical_judge"]
    }

def insight_synthesizer_reporter_node(state: AnalystGraphState) -> dict:
    """
    Part 5: The Insight Synthesizer & Reporter.
    Aggregates all state artifacts, historical node choices, model judgments,
    and critic validation findings into a production-grade automated markdown file.
    """
    print("\n [Node: Insight Synthesizer] Compiling final professional engineering analysis report...")
    
    # Bundle up our execution logs and structural statistics into a clean text packet
    technical_ledger = {
        "dataset_source": state.csv_path,
        "target_variable": state.target_column,
        "task_context": state.problem_type,
        "vision_structural_verdict": state.chosen_model_family,
        "final_model_metrics": state.final_model_metrics,
        "critic_validation_report": state.critic_validation_report,
        "critic_flags": state.critic_flags,
        "critic_diagnostic_plots": state.critic_plot_paths,
        "node_execution_sequence": state.execution_history
    }
    
    prompt = f"""
    You are a Principal Data Scientist and Technical Communicator. Review this internal system technical ledger:
    {json.dumps(technical_ledger, indent=2)}
    
    Generate a highly professional, beautifully structured Markdown Data Analysis Executive Summary Report.
    Use clear headings, clean markdown tables, bold highlights, and clean spacing. Include the following sections:
    
    ## 1. Executive Summary
    (Summarize what task context was discovered and what architecture was selected)
    
    ## 2. Structural Data Discovery
    (Outline the target variable inference and features)
    
    ## 3. Preprocessing & Model Selection Decisions
    (Detail exactly why the system routed the way it did, mentioning the Vision LLM's structural verdict)
    
    ## 4. Model Validation & Health Check
    (Detail the critic's findings: train vs test score gap, overfitting/underfitting verdict,
     any flags raised such as DATA_LEAKAGE_SUSPECTED or OVERFITTING_DETECTED.
     Reference the diagnostic plots generated: {list(state.critic_plot_paths.keys())}.
     Include the AI critic's natural language assessment.)
    
    ## 5. Pipeline Execution Audit Trail
    (Create a clean markdown table showing the progression trail of nodes: {state.execution_history})
    
    Write with authority, clarity, and analytical depth. Do not include any meta-commentary or conversational filler.
    """
    
    # Run our local text generator to draft the master summary
    final_report_markdown = query_local_text_llm(prompt)
    
    # Write the string out natively to your local hard drive file space
    dataset_name = os.path.basename(state.csv_path).split('.')[0]
    os.makedirs("reports", exist_ok=True)
    report_filename = f"reports/{dataset_name}_final_data_report.md"
    with open(report_filename, "w", encoding="utf-8") as file:
        file.write(final_report_markdown)
        
    print(f" Master Report successfully compiled and written to: '{report_filename}'")
    
    return {
        "execution_history": state.execution_history + ["insight_synthesizer_reporter"]
    }

def final_model_trainer_node(state: AnalystGraphState, 
                             X_train: pd.DataFrame, X_test: pd.DataFrame, 
                             y_train: pd.Series, y_test: pd.Series) -> dict:
    """
    Executes the Model Tournament & Hyperparameter Optimization.
    Instantiates a suite of algorithms based on the chosen geometric family (LINEAR or TREE),
    tunes them using RandomizedSearchCV to reduce error, and selects the absolute winner.
    """
    print(f"\n [Node: Model Tournament] Launching optimization suite for {state.problem_type.upper()} ({state.chosen_model_family} geometry)...")
    
    # 2. Define the Algorithm Suites & their Hyperparameter Grids
    # We use small randomized grids to keep local execution fast while still optimizing
    suite = []
    
    if state.problem_type == "regression":
        if state.chosen_model_family == "LINEAR":
            suite = [
                ("LinearRegression", LinearRegression(), {}),
                ("Ridge", Ridge(random_state=42), {'alpha': [0.1, 1.0, 10.0]}),
                ("Lasso", Lasso(random_state=42), {'alpha': [0.1, 1.0, 10.0]})
            ]
        else: # TREE / NON-LINEAR
            suite = [
                ("RandomForest", RandomForestRegressor(random_state=42), {'n_estimators': [50, 100], 'max_depth': [None, 10, 20]}),
                ("GradientBoosting", GradientBoostingRegressor(random_state=42), {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}),
                ("KNN", KNeighborsRegressor(), {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}),
                ("XGBoost", XGBRegressor(random_state=42, n_jobs=-1), {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}),
                ("CatBoost", CatBoostRegressor(random_state=42, verbose=0), {'iterations': [50, 100], 'learning_rate': [0.01, 0.1]}),
                ("LightGBM", LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1), {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]})
            ]
    else: # classification
        if state.chosen_model_family == "LINEAR":
            suite = [
                ("LogisticRegression", LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42), {'C': [0.1, 1.0, 10.0]}),
                ("LinearSVC", SVC(kernel='linear', class_weight='balanced', random_state=42), {'C': [0.1, 1.0, 10.0]})
            ]
        else: # TREE / NON-LINEAR
            suite = [
                ("RandomForest", RandomForestClassifier(class_weight='balanced', random_state=42), {'n_estimators': [50, 100], 'max_depth': [None, 10, 20]}),
                ("GradientBoosting", GradientBoostingClassifier(random_state=42), {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}),
                ("KNN", KNeighborsClassifier(), {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}),
                ("XGBoost", XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss'), {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}),
                ("CatBoost", CatBoostClassifier(random_state=42, verbose=0), {'iterations': [50, 100], 'learning_rate': [0.01, 0.1]}),
                ("LightGBM", LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1), {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]})
            ]
            
    # 3. Execute the Tournament
    best_model_name = ""
    best_model = None
    best_score = -float('inf')
    best_predictions = None
    
    scoring_metric = 'f1_macro' if state.problem_type == "classification" else 'r2'
    
    print(f" Optimizing {len(suite)} algorithms (Metric: {scoring_metric})...")
    
    for name, model, param_grid in suite:
        print(f"   -> Tuning {name}...")
        if not param_grid:
            # No hyperparameters to tune (e.g. standard LinearRegression)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = f1_score(y_test, preds, average='macro') if state.problem_type == "classification" else r2_score(y_test, preds)
            optimized_model = model
        else:
            search = RandomizedSearchCV(model, param_grid, n_iter=5, scoring=scoring_metric, cv=3, random_state=42, n_jobs=-1)
            search.fit(X_train, y_train)
            optimized_model = search.best_estimator_
            preds = optimized_model.predict(X_test)
            score = f1_score(y_test, preds, average='macro') if state.problem_type == "classification" else r2_score(y_test, preds)
            
        if score > best_score:
            best_score = score
            best_model = optimized_model
            best_model_name = name
            best_predictions = preds

    print(f"\n TOURNAMENT WINNER: {best_model_name} (Score: {best_score:.4f})")
    
    # 4. Extract Final Metrics for the Winner
    if state.problem_type == "regression":
        metrics = {
            "model_type": best_model_name,
            "r2_score": round(float(r2_score(y_test, best_predictions)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, best_predictions))), 4),
            "test_samples": int(len(X_test)),
            "train_samples": int(len(X_train))
        }
    else:
        metrics = {
            "model_type": best_model_name,
            "accuracy": round(float(accuracy_score(y_test, best_predictions)), 4),
            "macro_f1": round(float(f1_score(y_test, best_predictions, average='macro')), 4),
            "test_samples": int(len(X_test)),
            "train_samples": int(len(X_train))
        }
    
    print(f" Final Winner Metrics: {json.dumps(metrics, indent=2)}")
    
    # 5. Save predictions to CSV
    dataset_name = os.path.basename(state.csv_path).split('.')[0]
    os.makedirs("reports", exist_ok=True)
    output_path = f"reports/{dataset_name}_predictions_output.csv"
    pred_df = pd.DataFrame({
        "actual": y_test.values,
        "predicted": best_predictions
    })
    pred_df.to_csv(output_path, index=False)
    print(f" Predictions saved to: '{output_path}'")
    
    state_update = {
        "final_model_metrics": metrics,
        "prediction_output_path": output_path,
        "execution_history": state.execution_history + ["model_tournament_optimized"]
    }
    
    # Package model artifacts for the Critic/Validator Agent
    model_artifacts = {
        "best_model": best_model,
        "best_model_name": best_model_name,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "best_predictions": best_predictions
    }
    
    return state_update, model_artifacts

def critic_validator_node(state: AnalystGraphState, model_artifacts: dict) -> dict:
    """
    The Critic / Validator Agent.
    Runs Function 6 to compute diagnostic metrics and generate validation plots,
    then asks the local LLM to interpret the findings and produce a health verdict.
    """
    print("\n [Node: Critic Validator] Running production model health diagnostics...")
    
    # Execute the core diagnostic engine from engine_room
    critic_report = er.run_critic_validator(
        model=model_artifacts["best_model"],
        X_train=model_artifacts["X_train"],
        X_test=model_artifacts["X_test"],
        y_train=model_artifacts["y_train"],
        y_test=model_artifacts["y_test"],
        problem_type=state.problem_type
    )
    
    flags = critic_report["flags"]
    metrics = critic_report["metrics"]
    plot_paths = critic_report["plot_paths"]
    
    print(f" Critic Verdict: {metrics.get('verdict', 'N/A')}")
    print(f" Train Score: {metrics.get('train_score')} | Test Score: {metrics.get('test_score')} | Gap: {metrics.get('score_gap')}")
    if flags:
        print(f" ⚠ FLAGS RAISED: {flags}")
    print(f" Diagnostic Plots Generated: {list(plot_paths.keys())}")
    
    # Send findings to the local LLM for natural language interpretation
    interpretation_prompt = f"""
You are a Senior ML Model Critic. Analyze these diagnostic validation results from a {state.problem_type} model ({model_artifacts['best_model_name']}):

Train Score: {metrics.get('train_score')}
Test Score: {metrics.get('test_score')}
Score Gap: {metrics.get('score_gap')}
Verdict: {metrics.get('verdict')}
Flags: {flags if flags else 'None'}

Provide a concise 3-4 sentence technical assessment of the model's health.
- If OVERFITTING is detected, explain what is happening and suggest remedies (regularization, more data, simpler model).
- If UNDERFITTING is detected, suggest using a more complex model or engineering better features.
- If DATA_LEAKAGE is suspected, issue a strong warning to audit the preprocessing pipeline for target variable contamination.
- If HEALTHY, confirm the model generalizes well and the train/test gap is acceptable.

Keep it punchy, authoritative, and actionable. No filler.
"""
    
    llm_interpretation = query_local_text_llm(interpretation_prompt)
    print(f" AI Critic Assessment:\n{llm_interpretation}")
    
    # Store interpretation in the critic report for downstream use
    critic_report["metrics"]["llm_interpretation"] = llm_interpretation
    
    return {
        "critic_validation_report": critic_report["metrics"],
        "critic_flags": flags,
        "critic_plot_paths": plot_paths,
        "execution_history": state.execution_history + ["critic_validator"]
    }

def sandbox_code_generation_node(state: AnalystGraphState, X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Asks the local text LLM to generate a Python analysis script,
    executes it inside the sandbox, and retries with self-correction if it fails.
    Max 3 attempts before giving up gracefully.
    """
    print("\n [Node: Sandbox Code Generator] Requesting AI-generated analysis code...")
    
    base_prompt = f"""
You are an expert Python Data Scientist. Write a short Python analysis script for the following dataset context:
- Problem Type: {state.problem_type}
- Target Column: {state.target_column}
- Model Family Selected: {state.chosen_model_family}
- Available variables in scope: df (full combined DataFrame), X (features DataFrame), y (target Series), pd, np

Write Python code that:
1. Computes the correlation matrix between numeric features and prints the top 3 most correlated pairs
2. Prints the shape and basic statistics of X
3. Stores a dictionary with key findings in a variable called metrics_output

CRITICAL RULES:
- Do NOT import pandas or numpy (they are already available as pd and np)
- Do NOT read any CSV files
- Use only X and y variables directly
- Keep it under 20 lines
- You MUST wrap your entire response inside standard ```python fences.
"""
    
    dataset_name = os.path.basename(state.csv_path).split('.')[0]
    max_retries = 3
    sandbox_log = ""
    prompt = base_prompt
    
    for attempt in range(1, max_retries + 1):
        print(f" Sandbox Attempt {attempt}/{max_retries}...")
        
        generated_code = query_local_text_llm(prompt)
        
        # Robust code extraction
        import re
        code_blocks = re.findall(r'```python(.*?)```', generated_code, re.DOTALL)
        if code_blocks:
            generated_code = code_blocks[0].strip()
        elif "```" in generated_code:
            blocks = re.findall(r'```(.*?)```', generated_code, re.DOTALL)
            if blocks:
                generated_code = blocks[0].strip()
        else:
            # If no fences, assume the whole response is code but strip conversational filler
            lines = generated_code.split('\n')
            code_lines = [line for line in lines if not line.startswith(('Here', 'Sure', 'This', 'The'))]
            generated_code = '\n'.join(code_lines).strip()
        
        df_combined = pd.concat([X, y], axis=1)
        shared_context = {"X": X, "y": y, "df": df_combined, "pd": pd, "np": np}
        result = execute_sandboxed_code(generated_code, shared_context)
        
        if result["success"]:
            script_path = f"reports/{dataset_name}_sandbox_script.py"
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(generated_code)
                
            sandbox_log = f"Attempt {attempt} succeeded.\nLogs:\n{result['logs']}\nMetrics: {result['extracted_metrics']}"
            print(f" Sandbox code executed successfully on attempt {attempt}!")
            print(f" Generated Python script saved to: {script_path}")
            if result['logs']:
                print(f" Sandbox Output:\n{result['logs'].strip()}")
            break
        else:
            sandbox_log = f"Attempt {attempt} failed.\nError: {result['error_traceback']}"
            print(f" Attempt {attempt} failed. Routing traceback for self-correction...")
            
            # Append error context to the prompt for self-correction
            prompt = base_prompt + f"""

PREVIOUS ATTEMPT FAILED with this error:
{result['error_traceback']}

Fix the code and try again. Do not repeat the same mistake. Output ONLY raw Python code.
"""
    
    return {
        "sandbox_execution_log": sandbox_log,
        "execution_history": state.execution_history + ["sandbox_code_generator"]
    }

def dynamic_task_router(state: AnalystGraphState) -> str:
    """
    Deterministic routing switch checking target variable definitions.
    """
    print(f"[Backbone Router] Enforcing strict traffic management gateway for: {state.problem_type.upper()}")
    if state.problem_type == "regression":
        return "regression_pipeline"
    else:
        return "classification_pipeline"

if __name__ == "__main__":
    
    # 1. Structural Service Pre-Flight Check
    if "ERROR" in query_local_text_llm("ping"):
        print("\n RUNTIME ERROR: Please start your local Ollama engine first! (Run: 'ollama serve')")
        exit(1)
        
    print(" LOCAL DATA ANALYST AGENT WORKSPACE INITIALIZED")
    print("=====================================================================")
    
    # 2. Capture user file configuration path
    while True:
        user_csv = input("\n Enter the local path to your dataset (.csv): ").strip()
        # Clean quotes if user dragged and dropped the file into the terminal
        user_csv = user_csv.replace('"', '').replace("'", "")
        
        if os.path.exists(user_csv):
            print(f" Found data asset. Syncing runtime memory...")
            break
        print(" File path not found. Please verify the directory path and try again.")

    # 3. Instantiate the persistent Central Pydantic Graph State
    state = AnalystGraphState(csv_path=user_csv)
    
    import hashlib
    csv_hash = hashlib.md5(user_csv.encode('utf-8')).hexdigest()
    dataset_name = os.path.basename(user_csv).split('.')[0]
    os.makedirs("reports", exist_ok=True)
    cache_file = f"reports/{dataset_name}_pipeline_cache_{csv_hash}.json"
    
    use_cache = False
    if os.path.exists(cache_file):
        ans = input(f"\n [Cache Found] An existing trained pipeline was found for this dataset. Load it and skip training? (y/n): ").strip().lower()
        if ans == 'y':
            with open(cache_file, "r") as f:
                state = AnalystGraphState(**json.load(f))
            use_cache = True
            
            print(" Fast-forwarding: Reconstructing data geometry...")
            X_train_clean, X_test_clean, y_train_clean, y_test_clean, _ = er.global_preprocess(state.csv_path, target_column=state.target_column)
            if state.chosen_model_family == "LINEAR":
                X_train_final, X_test_final, y_train_final = er.model_specific_optimizer(
                    X_train_clean, X_test_clean, y_train_clean, y_test_clean
                )
                y_test_final = y_test_clean
            else:
                X_train_final, X_test_final = X_train_clean, X_test_clean
                y_train_final, y_test_final = y_train_clean, y_test_clean
            X_final, y_final = X_train_final, y_train_final

    if not use_cache:
        # 4. Run Core Pipeline Preparation (Parts 1 & 2 structural analysis)
        state = state.model_copy(update=data_scout_node(state))
        state = state.model_copy(update=target_inference_node(state))
        state = state.model_copy(update=global_preprocessor_node(state))
        
        # 5. Determine the task management routing lane
        routing_lane = dynamic_task_router(state)
        
        # 6. Get preprocessed data for all downstream nodes (single read)
        X_train_clean, X_test_clean, y_train_clean, y_test_clean, problem_type = er.global_preprocess(state.csv_path, target_column=state.target_column)
        
        # 7. Run branch-specific evaluation layers
        if routing_lane == "regression_pipeline":
            print("\n REGRESSION BRANCH: Running OLS baseline & visual judge...")
            if state.problem_type == "regression":
                dataset_name = os.path.basename(state.csv_path).split('.')[0]
                metrics_dict = er.run_baseline_ols(X_train_clean, y_train_clean, output_plot_path=f"plots/{dataset_name}_residual_plot.png")
                state = state.model_copy(update={"regression_metrics": metrics_dict})
                state = state.model_copy(update=regression_visual_judge_node(state))
            
        elif routing_lane == "classification_pipeline":
            print("\n CLASSIFICATION BRANCH: Running classifier benchmarker & statistical judge...")
            benchmarks = er.run_classifier_benchmarker(X_train_clean, y_train_clean)
            state = state.model_copy(update={"classification_benchmarks": benchmarks})
            state = state.model_copy(update=classification_statistical_judge_node(state))
        
        # 8. Optimizer: Conditionally apply outlier removal + robust scaling for LINEAR models
        if state.chosen_model_family == "LINEAR":
            print("\n [Optimizer] LINEAR model selected — applying outlier removal & robust scaling...")
            X_train_opt, X_test_opt, y_train_opt = er.model_specific_optimizer(X_train_clean, X_test_clean, y_train_clean, y_test_clean)
            rows_removed = len(X_train_clean) - len(X_train_opt)
            print(f" Outlier surgery complete: {rows_removed} rows removed from training set, {len(X_train_opt)} rows retained.")
            state = state.model_copy(update={
                "outliers_detected": rows_removed > 0,
                "execution_history": state.execution_history + ["optimizer_applied"]
            })
            X_train_final, X_test_final = X_train_opt, X_test_opt
            y_train_final, y_test_final = y_train_opt, y_test_clean
        else:
            print("\n [Optimizer] TREE model selected — skipping optimization (scale/outlier invariant).")
            state = state.model_copy(update={
                "execution_history": state.execution_history + ["optimizer_skipped"]
            })
            X_train_final, X_test_final = X_train_clean, X_test_clean
            y_train_final, y_test_final = y_train_clean, y_test_clean
        
        X_final, y_final = X_train_final, y_train_final
        
        # 9. Train the final production model and save predictions to CSV
        trainer_state_update, model_artifacts = final_model_trainer_node(state, X_train_final, X_test_final, y_train_final, y_test_final)
        state = state.model_copy(update=trainer_state_update)
        
        # 9b. Run the Critic/Validator Agent to diagnose model health
        state = state.model_copy(update=critic_validator_node(state, model_artifacts))
        
        # 10. Run the sandbox code generation node for AI-generated exploratory analysis
        state = state.model_copy(update=sandbox_code_generation_node(state, X_final, y_final))
    
        # 11. Generate the final Executive Summary Markdown Report
        state = state.model_copy(update=insight_synthesizer_reporter_node(state))
        
        with open(cache_file, "w") as f:
            f.write(state.model_dump_json(indent=2))
        print(f"\n [System] Pipeline state securely cached to disk for fast-resuming.")
    
    # 12. Build data sample context for Q&A grounding
    raw_df = pd.read_csv(user_csv)
    data_sample = (
        f"Dataset Head (first 5 rows):\n{raw_df.head(5).to_string()}\n\n"
        f"Statistical Summary:\n{raw_df.describe().to_string()}\n\n"
        f"Column Types:\n{raw_df.dtypes.to_string()}\n\n"
        f"Shape: {raw_df.shape[0]} rows x {raw_df.shape[1]} columns"
    )
    state = state.model_copy(update={"data_sample_context": data_sample})
    
    print("\n FULL PIPELINE COMPLETE. READY FOR ANALYSIS QUESTIONS.")
    print(f" Final Model: {state.final_model_metrics.get('model_type', 'N/A')}")
    print(f" Predictions saved to: {state.prediction_output_path}")
    print(" Type your questions about the dataset below. Type 'exit' to quit.")
    print("---------------------------------------------------------------------")

    # 12. Continuous Interactive Conversation Workspace Loop (Data-Aware Q&A)
    while True:
        try:
            user_question = input("\n Ask a question about your data: ").strip()
            
            if user_question.lower() in ["exit", "quit", "q"]:
                print("\n Closing local analyst workspace session. Happy engineering!")
                break
                
            if not user_question:
                continue
                
            print(" Thinking...")
            
            dataset_name = os.path.basename(state.csv_path).split('.')[0]
            # Formulate a prompt grounded in both pipeline metadata AND actual data
            analytical_prompt = f"""
You are an expert Data Science Companion. You are interacting with a user regarding their dataset.

[CRITICAL CONTEXT LEDGER]
- Dataset Source File  : {state.csv_path}
- Detected Target     : {state.target_column}
- Mathematical Profile : {state.problem_type}
- Model Family Verdict : {state.chosen_model_family}
- Outliers Detected    : {state.outliers_detected}
- Final Model Metrics  : {json.dumps(state.final_model_metrics)}
- Baseline Statistics  : {json.dumps(state.regression_metrics if state.problem_type == 'regression' else state.classification_benchmarks)}
- Predictions File     : {state.prediction_output_path}
- Pipeline Trail       : {state.execution_history}

[ACTUAL DATA SNAPSHOT]
{state.data_sample_context}

User's Direct Question: {user_question}

Provide a precise, analytically grounded answer based strictly on the metrics ledger and data snapshot provided above.
If the user asks to plot, visualize, or execute something, generate a python script enclosed in ```python fences.
The script has access to `df` (full DataFrame with both features and target), `X` (features DataFrame), `y` (target Series), `pd`, `np`, `plt`, `sns`, and `feature_importances` (a dictionary mapping features to their importance score).
CRITICAL: Use `plt.savefig('plots/{dataset_name}_custom_plot.png')` to save any plots instead of `plt.show()`.
CRITICAL: Do NOT try to read the CSV file using pd.read_csv. Use ONLY the variables explicitly provided above! Do NOT assume any other variables exist.
Do not guess information outside this profile schema boundaries. Keep it punchy and professional.
"""
            
            answer = query_local_text_llm(analytical_prompt)
            
            # Dynamic Execution Gateway (If the Agent wrote code, run it!)
            if "```" in answer:
                import re
                code_blocks = re.findall(r'```python(.*?)```', answer, re.DOTALL)
                if not code_blocks:
                    code_blocks = re.findall(r'```(.*?)```', answer, re.DOTALL)
                
                if code_blocks:
                    code_to_run = code_blocks[0].strip()
                    script_path = f"reports/{dataset_name}_qa_script.py"
                    
                    # Clean the answer text to hide the raw code and substitute a message
                    clean_answer = re.sub(r'```.*?```', f'\n[Python script automatically saved to {script_path}]\n', answer, flags=re.DOTALL)
                    print(f"\n Agent:\n{clean_answer.strip()}")
                    print("-" * 69)
                    
                    df_combined = pd.concat([X_final, y_final], axis=1)
                    
                    # Provide extra ledger metrics directly to the context to prevent hallucinated variables
                    feature_importances = state.classification_benchmarks.get("feature_importances", {}) if state.problem_type == "classification" else {}
                    shared_context = {
                        "X": X_final, "y": y_final, "df": df_combined, "pd": pd, "np": np, 
                        "plt": plt, "sns": sns, "feature_importances": feature_importances
                    }

                    # Add self-correction retry loop for QA code
                    max_retries = 3
                    for attempt in range(1, max_retries + 1):
                        with open(script_path, "w", encoding="utf-8") as f:
                            f.write(code_to_run)
                            
                        print(f" [Sandbox] Running the generated script (Attempt {attempt})...")
                        result = execute_sandboxed_code(code_to_run, shared_context)
                        
                        if result["success"]:
                            print(" [Sandbox] Code executed successfully!")
                            if "savefig" in code_to_run:
                                print(" [Sandbox] Plot should be saved to your 'plots/' directory.")
                            break
                        else:
                            print(f" [Sandbox] Error executing code on attempt {attempt}:\n{result['error_traceback']}")
                            if attempt < max_retries:
                                print(" [Sandbox] Asking AI to self-correct the code...")
                                correction_prompt = analytical_prompt + f"\n\nYour previous code failed with this error:\n{result['error_traceback']}\n\nPlease fix the code and try again. Output ONLY the raw Python code enclosed in ```python fences."
                                correction_answer = query_local_text_llm(correction_prompt)
                                correction_blocks = re.findall(r'```python(.*?)```', correction_answer, re.DOTALL)
                                if not correction_blocks:
                                    correction_blocks = re.findall(r'```(.*?)```', correction_answer, re.DOTALL)
                                if correction_blocks:
                                    code_to_run = correction_blocks[0].strip()
                                else:
                                    break # Give up if no code block found
                else:
                    print(f"\n Agent:\n{answer.strip()}")
                    print("-" * 69)
            else:
                print(f"\n Agent:\n{answer.strip()}")
                print("-" * 69)
            
        except KeyboardInterrupt:
            print("\n Session interrupted via terminal control wrapper.")
            break