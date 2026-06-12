import os
import pandas as pd
import numpy as np 
import statsmodels.api as sm  
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import seaborn as sns 
import json 
from sklearn.ensemble import IsolationForest 
from sklearn.preprocessing import RobustScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# Function 1: The Metadata Extractor
def extract_metadata(file_path: str) -> str:
    """
    Loads a CSV file safely and extracts structural metadata.
    Returns a small JSON string to save local LLM VRAM context.
    """
    df = pd.read_csv(file_path)

    metadata = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": {}
    }

    for col in df.columns:
        unique_count = int(df[col].nunique())
        missing_count = int(df[col].isnull().sum())
        data_type = str(df[col].dtype)

        col_info = {
            "data_type": data_type,
            "missing_values": missing_count,
            "cardinality": unique_count
        }

        # Give the LLM sample values for context if it's a low-cardinality categorical
        if unique_count <=10:
            col_info["sample_values"] = [str(x) for x in df[col].dropna().unique()[:5]]

        metadata["columns"][col] = col_info

    return json.dumps(metadata, indent = 4)

# Function 2: The automated preprocessor

def global_preprocess(file_path: str, target_column: str, cat_threshold: int = 10) -> tuple:
    """
    Enforces structural integrity for all downstream models.
    Splits the data into train/test sets first to prevent target leakage,
    then applies imputations and encodings fitted only on the training set.
    
    Returns: (X_train_clean, X_test_clean, y_train_clean, y_test_clean, problem_type)
    """
    df = pd.read_csv(file_path)
    
    # Sanitize column names: XGBoost and LightGBM crash if columns have spaces or JSON characters []{}<>,
    import re
    df.columns = [re.sub(r'[^\w\s]', '', col).strip().replace(' ', '_') for col in df.columns]
    
    # Since we changed column names, we must ensure the target_column name matches if it had spaces
    target_column = re.sub(r'[^\w\s]', '', target_column).strip().replace(' ', '_')
    
    # Structural Check: Ensure target column isn't missing completely
    df = df.dropna(subset=[target_column])

    # Extract target and features
    y = df[target_column]
    X = df.drop(columns = [target_column])

    # Identify and drop numeric ID-like columns (high cardinality numeric columns)
    num_cols_raw = X.select_dtypes(include=[np.number]).columns
    total_rows = len(X)
    numeric_ids_to_drop = []
    for col in num_cols_raw:
        if X[col].nunique() >= total_rows * 0.4:
            numeric_ids_to_drop.append(col)
    if numeric_ids_to_drop:
        print(f"   [Preprocessor] Dropping numeric ID-like columns: {numeric_ids_to_drop}")
        X = X.drop(columns=numeric_ids_to_drop)

# 1. Dynamic Problem Type detection and Target Processing
    # Check if the target is text (object/category) OR numerical but acts as discrete classes
    if y.dtype == 'object' or y.dtype == 'category' or y.nunique() <= 10:
        problem_type = "classification"
        
        # Safe Label Encoding: Converts text classes to sequential integers (0, 1, 2... N)
        # Standard classification models treat these as pure IDs, not numerical scales.
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y.astype(str)), name=target_column, index=df.index)
    else:
        problem_type = "regression"
        # Keep y as its organic continuous numerical values

    # 2. Split the dataset FIRST to prevent data leakage
    stratify_param = y if problem_type == "classification" and y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_param
    )

    # 3. Separate numerical and categorical columns for features
    num_cols = X_train.select_dtypes(include = [np.number]).columns
    cat_cols = X_train.select_dtypes(exclude = [np.number]).columns

    # 4. Handle missing values (Imputation fitted only on train)
    for col in num_cols:
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)
    for col in cat_cols:
        mode_val = X_train[col].mode()[0] if not X_train[col].empty else "missing"
        X_train[col] = X_train[col].fillna(mode_val)
        X_test[col] = X_test[col].fillna(mode_val)

    # 4.5 PRE-ENCODING LEAKAGE GUARD — Detect categorical columns that perfectly partition the target
    # A categorical column is leaky if ANY of its values maps to the target with 100% purity.
    # E.g., fraud_risk='High' → fraud_flag=1 always, fraud_risk='Low' → fraud_flag=0 always.
    # This must run BEFORE encoding to catch these columns before they enter the model.
    leaky_cat_cols = []
    for col in list(cat_cols):
        try:
            # For each unique value, check if it maps to a single target class
            grouped = y_train.groupby(X_train[col])
            has_pure_partition = False
            for val, group_y in grouped:
                if len(group_y) >= 10 and group_y.nunique() == 1:
                    # This category value perfectly predicts one class
                    has_pure_partition = True
                    break
            if has_pure_partition:
                leaky_cat_cols.append(col)
        except Exception:
            pass
    
    if leaky_cat_cols:
        print(f"\n   [WARNING] [LEAKAGE GUARD] Categorical columns with perfect target partitioning detected:")
        for col in leaky_cat_cols:
            unique_vals = X_train[col].unique()[:5]
            print(f"     --> '{col}' (values: {list(unique_vals)}) -- DROPPING (contains label-leaking categories)")
        X_train = X_train.drop(columns=leaky_cat_cols)
        X_test = X_test.drop(columns=leaky_cat_cols)
        # Update cat_cols list after dropping
        cat_cols = X_train.select_dtypes(exclude=[np.number]).columns

    # 4.6 DUPLICATE COLUMN DETECTOR — Remove columns that are 100% identical to another
    num_cols_current = X_train.select_dtypes(include=[np.number]).columns
    duplicate_cols = []
    seen_cols = []
    for i, col1 in enumerate(num_cols_current):
        for col2 in seen_cols:
            if (X_train[col1] == X_train[col2]).all():
                duplicate_cols.append((col1, col2))
                break
        else:
            seen_cols.append(col1)
    
    if duplicate_cols:
        cols_to_drop = [pair[0] for pair in duplicate_cols]
        print(f"\n   [WARNING] [DUPLICATE GUARD] Removing {len(cols_to_drop)} duplicate columns:")
        for dup, orig in duplicate_cols:
            print(f"     --> '{dup}' is identical to '{orig}' -- DROPPING")
        X_train = X_train.drop(columns=cols_to_drop)
        X_test = X_test.drop(columns=cols_to_drop)

    # Snapshot binary columns BEFORE encoding (to distinguish original flags from one-hot dummies)
    pre_encoding_binary_cols = []
    for col in X_train.select_dtypes(include=[np.number]).columns:
        if X_train[col].nunique() == 2 and set(X_train[col].unique()).issubset({0, 1, 0.0, 1.0}):
            pre_encoding_binary_cols.append(col)

    # 5. Adaptive categorical encoding pipeline (fitted only on train)
    total_train_rows = len(X_train)
    # Re-fetch cat_cols after potential drops
    cat_cols = X_train.select_dtypes(exclude=[np.number]).columns
    for col in cat_cols:
        unique_count = X_train[col].nunique()

        if unique_count >= total_train_rows * 0.4:
            # Drop it. If 40%+ of training rows are unique, it's an ID, not a feature.
            X_train = X_train.drop(columns=[col])
            X_test = X_test.drop(columns=[col])
        elif unique_count <= cat_threshold:
            # Low Cardinality -> Safe One-Hot Encoding
            dummies_train = pd.get_dummies(X_train[col], prefix=col, drop_first=True)
            dummies_test = pd.get_dummies(X_test[col], prefix=col, drop_first=True)
            # Align test dummies to train dummies (fill missing columns with 0)
            dummies_test = dummies_test.reindex(columns=dummies_train.columns, fill_value=0)
            
            X_train = pd.concat([X_train.drop(columns=[col]), dummies_train], axis=1)
            X_test = pd.concat([X_test.drop(columns=[col]), dummies_test], axis=1)
        else:
            # Medium Cardinality -> Frequency Encoding (leakage-free)
            # Encodes each category as its frequency (count / total_rows) in the training set.
            # This captures rarity/prevalence WITHOUT referencing the target variable y.
            freq_map = X_train[col].value_counts(normalize=True)
            X_train[f"{col}_freq_enc"] = X_train[col].map(freq_map)
            X_test[f"{col}_freq_enc"] = X_test[col].map(freq_map)
            # Impute any unseen categories in test set with a small epsilon frequency
            X_test[f"{col}_freq_enc"] = X_test[f"{col}_freq_enc"].fillna(0.0)
            
            X_train = X_train.drop(columns=[col])
            X_test = X_test.drop(columns=[col])

    # Ensure all boolean/categorical flags are standard floats for scikit-learn compatibility
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)

    # 6. LEAKY FEATURE DETECTOR (Two-Phase)
    # Phase A: Drop columns with |correlation| >= 0.95 (direct numerical proxies)
    leaky_cols_corr = []
    for col in X_train.columns:
        try:
            corr = X_train[col].corr(y_train)
            if abs(corr) >= 0.95:
                leaky_cols_corr.append((col, round(corr, 4)))
        except Exception:
            pass
    
    if leaky_cols_corr:
        print(f"\n   [WARNING] [LEAKAGE GUARD Phase A] High-correlation features detected:")
        for col_name, corr_val in leaky_cols_corr:
            print(f"     --> '{col_name}' (corr = {corr_val}) -- DROPPING")
        drop_names = [c[0] for c in leaky_cols_corr]
        X_train = X_train.drop(columns=drop_names)
        X_test = X_test.drop(columns=drop_names)
    
    # Phase B: Detect binary flag columns that are derived from the target
    # In synthetic datasets, columns like 'unusual_amount_flag' or 'previous_fraud_flag'
    # are often reverse-engineered from the target. Each one has moderate AUC (~0.55-0.71)
    # but TOGETHER they perfectly reconstruct the target.
    # Strategy: Collect all binary (0/1) columns, train a quick model on ONLY those flags,
    # and if that subset alone achieves F1 >= 0.95, they are collectively leaking.
    from sklearn.metrics import f1_score as f1_check
    
    binary_flag_cols = [col for col in pre_encoding_binary_cols if col in X_train.columns]
    
    leaky_flags_dropped = False
    if len(binary_flag_cols) >= 3:
        # Quick multivariate leakage test: can the binary flags alone predict the target?
        from sklearn.model_selection import train_test_split as tts_check
        X_flags_tr, X_flags_val, y_flags_tr, y_flags_val = tts_check(
            X_train[binary_flag_cols], y_train, test_size=0.2, random_state=42,
            stratify=y_train if y_train.nunique() > 1 else None
        )
        # Use a small RandomForest (not LogReg) because flag leakage can be non-linear
        quick_model = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42, n_jobs=-1)
        quick_model.fit(X_flags_tr, y_flags_tr)
        quick_preds = quick_model.predict(X_flags_val)
        flag_f1 = float(f1_check(y_flags_val, quick_preds, average='macro'))
        
        if flag_f1 >= 0.90:
            print(f"\n   [WARNING] [LEAKAGE GUARD Phase B] Binary flag columns collectively predict target with F1 = {flag_f1:.4f}!")
            print(f"     These {len(binary_flag_cols)} flags appear to be derived from the target label:")
            for col in binary_flag_cols:
                print(f"     --> '{col}' -- DROPPING")
            X_train = X_train.drop(columns=binary_flag_cols)
            X_test = X_test.drop(columns=binary_flag_cols)
            leaky_flags_dropped = True
        else:
            print(f"\n   [OK] [LEAKAGE GUARD Phase B] Binary flag group F1 = {flag_f1:.4f} -- below leakage threshold. Keeping flags.")
    
    total_dropped = len(leaky_cols_corr) + (len(binary_flag_cols) if leaky_flags_dropped else 0)
    if total_dropped > 0:
        print(f"   [OK] {total_dropped} leaky features removed. {len(X_train.columns)} features remain.")

    return X_train, X_test, y_train, y_test, problem_type

# Function 3: The baseline OLS & Residual plotter

def run_baseline_ols(X: pd.DataFrame, y: pd.Series, output_plot_path: str = "plots/residual_plot.png") -> dict:
    """
    Fits an OLS baseline regression model to generate detailed statistical tables
    and exports a visual diagnostic residual plot for the future Vision Agent.
    """
    # Statsmodels requires an explicit intercept constant column added
    X_constant = sm.add_constant(X)
    
    model = sm.OLS(y, X_constant).fit()
    
    residuals = model.resid
    fitted_values = model.fittedvalues
    
    # Output Residual Plot
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=fitted_values, y=residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title("OLS Residuals vs Predicted")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.tight_layout()
    plt.savefig(output_plot_path)
    plt.close() # Free local system memory
    
    # Compile a dictionary containing rich statistical metrics for the Text LLM
    ols_summary = {
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "f_statistic": float(model.fvalue),
        "f_pvalue": float(model.f_pvalue),
        "residual_plot_path": output_plot_path
    }
    
    return ols_summary

# Function 4: The model specific optimizer
def model_specific_optimizer(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                             y_train: pd.Series, y_test: pd.Series,
                             strategy: str = "iqr") -> tuple:
    """
    Precision Data Surgery tool. Applies outlier filtering and Robust Scaling.
    Only triggered conditionally if a distance-sensitive model is chosen.
    
    IMPORTANT: Fits IQR bounds and scaler on TRAINING data only, then transforms
    both train and test to prevent data leakage.
    
    Returns: (X_train_scaled, X_test_scaled, y_train_filtered)
    """
    X_tr = X_train.copy()
    y_tr = y_train.copy()
    X_te = X_test.copy()

    # 1. Outlier removal surgery (applied only to training set)
    if strategy == "iqr":
        # Compute IQR bounds on training data, filter training rows only
        for col in X_tr.columns:
            q1 = X_tr[col].quantile(0.25)
            q3 = X_tr[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # Only remove outliers from TRAINING set (test set stays untouched)
            mask = (X_tr[col] >= lower_bound) & (X_tr[col] <= upper_bound)
            X_tr = X_tr[mask]
            y_tr = y_tr[mask]

    elif strategy == "isolation_forest":
        # Multivariate Anomaly Detection (fit on train only)
        iso = IsolationForest(contamination=0.05, random_state=42)
        preds = iso.fit_predict(X_tr)
        mask = preds == 1
        X_tr = X_tr[mask]
        y_tr = y_tr[mask]

    # 2. Feature Scaling using RobustScaler (fit on train, transform both)
    scaler = RobustScaler()
    X_tr_scaled = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index)
    X_te_scaled = pd.DataFrame(scaler.transform(X_te), columns=X_te.columns, index=X_te.index)
    
    return X_tr_scaled, X_te_scaled, y_tr

#Function 5: The classifier Benchmark
def run_classifier_benchmarker(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Benchmarks a Linear vs Non-Linear classifier using cross-validation
    on the training set. Returns Macro F1-Scores to let the agent choose the optimal path.
    
    NOTE: Uses cross_val_score instead of a second train_test_split to avoid
    distributional bias from double-splitting already-split data.
    """
    from sklearn.model_selection import cross_val_score
    
    # 1. Cross-validated Linear Baseline (Logistic Regression)
    linear_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    linear_cv_scores = cross_val_score(linear_model, X, y, cv=3, scoring='f1_macro', n_jobs=-1)
    linear_f1 = float(linear_cv_scores.mean())
    
    # 2. Cross-validated Non-Linear Challenger (Random Forest)
    tree_model = RandomForestClassifier(class_weight='balanced', random_state=42)
    tree_cv_scores = cross_val_score(tree_model, X, y, cv=3, scoring='f1_macro', n_jobs=-1)
    tree_f1 = float(tree_cv_scores.mean())
    
    # 3. Extract Feature Importances from the Tree Model (fit on full training set)
    tree_model.fit(X, y)
    importances = tree_model.feature_importances_
    feature_importance_dict = {
        col: float(imp) for col, imp in zip(X.columns, importances)
    }
    # Sort feature importance from highest to lowest
    sorted_importances = dict(sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True))
    
    # 4. Package results for the Central Graph State
    benchmarks = {
        "linear_baseline_f1": linear_f1,
        "tree_challenger_f1": tree_f1,
        "delta": float(tree_f1 - linear_f1),
        "feature_importances": sorted_importances
    }
    
    return benchmarks

# Function 6: The Critic / Validator Agent Engine

def run_critic_validator(model, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                         y_train: pd.Series, y_test: pd.Series, 
                         problem_type: str, dataset_name: str = "dataset") -> dict:
    """
    Production-grade model validation engine. Computes train vs test score gaps,
    generates diagnostic plots (confusion matrix, ROC-AUC, PR curve, learning curve),
    and flags overfitting, underfitting, or data leakage for the Critic Agent.
    
    Returns: dict with keys: metrics, flags, plot_paths, classification_report_dict
    """
    from sklearn.metrics import (confusion_matrix, classification_report, 
                                 roc_curve, auc, precision_recall_curve, 
                                 average_precision_score)
    from sklearn.model_selection import learning_curve
    from sklearn.preprocessing import label_binarize
    
    flags = []
    plot_paths = {}
    critic_metrics = {}
    class_report_dict = {}
    
    # ========================================================================
    # 1. TRAIN vs TEST SCORE GAP ANALYSIS (Overfitting / Underfitting / Leakage)
    # ========================================================================
    if problem_type == "classification":
        train_score = float(model.score(X_train, y_train))  # Accuracy on training data
        test_score = float(model.score(X_test, y_test))      # Accuracy on test data
    else:
        train_score = float(model.score(X_train, y_train))  # R² on training data
        test_score = float(model.score(X_test, y_test))      # R² on test data
    
    score_gap = round(train_score - test_score, 4)
    
    critic_metrics["train_score"] = round(train_score, 4)
    critic_metrics["test_score"] = round(test_score, 4)
    critic_metrics["score_gap"] = score_gap
    
    # Diagnostic Flag Logic
    if train_score >= 0.99 and test_score >= 0.99:
        # Both train AND test are near-perfect → classic target leakage signature
        flags.append("DATA_LEAKAGE_SUSPECTED")
        critic_metrics["verdict"] = "DATA_LEAKAGE_SUSPECTED"
    elif train_score >= 0.99 and score_gap > 0.15:
        # Train is perfect but test is much lower → leakage or severe overfitting
        flags.append("DATA_LEAKAGE_SUSPECTED")
        critic_metrics["verdict"] = "DATA_LEAKAGE_SUSPECTED"
    elif score_gap > 0.15:
        flags.append("OVERFITTING_DETECTED")
        critic_metrics["verdict"] = "OVERFITTING"
    elif train_score < 0.5 and test_score < 0.5:
        flags.append("UNDERFITTING_DETECTED")
        critic_metrics["verdict"] = "UNDERFITTING"
    else:
        critic_metrics["verdict"] = "HEALTHY"
    
    predictions = model.predict(X_test)
    
    # ========================================================================
    # 2. CONFUSION MATRIX (Classification Only)
    # ========================================================================
    if problem_type == "classification":
        cm = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=sorted(y_test.unique()),
                    yticklabels=sorted(y_test.unique()))
        plt.title("Confusion Matrix — Model Diagnostic")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        os.makedirs("plots", exist_ok=True)
        cm_path = f"plots/{dataset_name}_confusion_matrix.png"
        plt.savefig(cm_path, dpi=150)
        plt.close()
        plot_paths["confusion_matrix"] = cm_path
        
        # Per-class classification report
        class_report_dict = classification_report(y_test, predictions, output_dict=True, zero_division=0)
        critic_metrics["classification_report"] = class_report_dict
    
    # ========================================================================
    # 3. ROC-AUC CURVE (Classification Only — requires predict_proba)
    # ========================================================================
    if problem_type == "classification":
        has_proba = hasattr(model, "predict_proba")
        
        if has_proba:
            classes = sorted(y_test.unique())
            n_classes = len(classes)
            
            if n_classes == 2:
                # Binary classification: single ROC curve
                y_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc_score_val = auc(fpr, tpr)
                
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2, 
                         label=f'ROC Curve (AUC = {roc_auc_score_val:.4f})')
                plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Baseline')
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC-AUC Curve — Binary Classification")
                plt.legend(loc="lower right")
                plt.tight_layout()
                os.makedirs("plots", exist_ok=True)
                roc_path = f"plots/{dataset_name}_roc_auc_curve.png"
                plt.savefig(roc_path, dpi=150)
                plt.close()
                plot_paths["roc_auc_curve"] = roc_path
                critic_metrics["roc_auc"] = round(roc_auc_score_val, 4)
            else:
                # Multi-class: One-vs-Rest ROC curves
                y_proba = model.predict_proba(X_test)
                y_test_bin = label_binarize(y_test, classes=classes)
                
                plt.figure(figsize=(8, 6))
                auc_scores = {}
                for i, cls in enumerate(classes):
                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                    roc_auc_val = auc(fpr, tpr)
                    auc_scores[f"class_{cls}"] = round(roc_auc_val, 4)
                    plt.plot(fpr, tpr, lw=2, label=f'Class {cls} (AUC = {roc_auc_val:.4f})')
                
                plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Baseline')
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC-AUC Curves — Multi-Class (One-vs-Rest)")
                plt.legend(loc="lower right", fontsize=8)
                plt.tight_layout()
                os.makedirs("plots", exist_ok=True)
                roc_path = f"plots/{dataset_name}_roc_auc_curve.png"
                plt.savefig(roc_path, dpi=150)
                plt.close()
                plot_paths["roc_auc_curve"] = roc_path
                critic_metrics["roc_auc_per_class"] = auc_scores
        else:
            critic_metrics["roc_auc"] = "SKIPPED — model does not support predict_proba"
    
    # ========================================================================
    # 4. PRECISION-RECALL CURVE (Classification Only — requires predict_proba)
    # ========================================================================
    if problem_type == "classification" and has_proba:
        classes = sorted(y_test.unique())
        n_classes = len(classes)
        
        if n_classes == 2:
            y_proba = model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            avg_precision = average_precision_score(y_test, y_proba)
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='darkorange', lw=2, 
                     label=f'PR Curve (AP = {avg_precision:.4f})')
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision-Recall Curve — Binary Classification")
            plt.legend(loc="lower left")
            plt.tight_layout()
            os.makedirs("plots", exist_ok=True)
            pr_path = f"plots/{dataset_name}_precision_recall_curve.png"
            plt.savefig(pr_path, dpi=150)
            plt.close()
            plot_paths["precision_recall_curve"] = pr_path
            critic_metrics["average_precision"] = round(avg_precision, 4)
        else:
            y_proba = model.predict_proba(X_test)
            y_test_bin = label_binarize(y_test, classes=classes)
            
            plt.figure(figsize=(8, 6))
            ap_scores = {}
            for i, cls in enumerate(classes):
                precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_proba[:, i])
                avg_prec = average_precision_score(y_test_bin[:, i], y_proba[:, i])
                ap_scores[f"class_{cls}"] = round(avg_prec, 4)
                plt.plot(recall, precision, lw=2, label=f'Class {cls} (AP = {avg_prec:.4f})')
            
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision-Recall Curves — Multi-Class (One-vs-Rest)")
            plt.legend(loc="lower left", fontsize=8)
            plt.tight_layout()
            os.makedirs("plots", exist_ok=True)
            pr_path = f"plots/{dataset_name}_precision_recall_curve.png"
            plt.savefig(pr_path, dpi=150)
            plt.close()
            plot_paths["precision_recall_curve"] = pr_path
            critic_metrics["average_precision_per_class"] = ap_scores
    
    # ========================================================================
    # 5. LEARNING CURVE (Both Regression & Classification)
    # ========================================================================
    scoring = 'f1_macro' if problem_type == "classification" else 'r2'
    
    try:
        train_sizes, train_scores, val_scores = learning_curve(
            model, 
            X_train,    # Use ONLY training data — never include test set
            y_train,
            train_sizes=np.linspace(0.2, 1.0, 5),
            cv=3,
            scoring=scoring,
            n_jobs=-1,
            random_state=42
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(8, 6))
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='red')
        plt.title(f"Learning Curve — {type(model).__name__}")
        plt.xlabel("Training Set Size")
        plt.ylabel(f"Score ({scoring})")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        os.makedirs("plots", exist_ok=True)
        lc_path = f"plots/{dataset_name}_learning_curve.png"
        plt.savefig(lc_path, dpi=150)
        plt.close()
        plot_paths["learning_curve"] = lc_path
    except Exception as e:
        critic_metrics["learning_curve_error"] = str(e)
    
    # ========================================================================
    # 6. COMPILE FINAL CRITIC REPORT
    # ========================================================================
    return {
        "metrics": critic_metrics,
        "flags": flags,
        "plot_paths": plot_paths,
        "classification_report": class_report_dict
    }
