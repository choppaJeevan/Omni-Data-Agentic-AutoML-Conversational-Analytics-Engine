from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any 

class AnalystGraphState(BaseModel):
    """
    The unified, type-safe central memory space for the local AI Data Agent.
    Tracks structural metadata, algorithmic pathways, evaluation metrics, and error traces.
    """
    # File & Path Tracking
    csv_path: str = Field(description = "system path to the raw input dataset CSV file.")
    residual_plot_path: Optional[str] = Field(default = "residual_plot.png", description = "Path to the generated diagnostic residual plot image.")

    # Target Variable Information
    target_column: Optional[str] = Field(default = None, description = "The identified column name representing the ML target variable.")
    problem_type: Optional[str] = Field(default = None, description = "Determined problem context: 'Regression' or 'Classification'.")

    # Engine Room artifacts (JSON summary from Function 1)
    metadata_summary: Optional[Dict[str, Any]] = Field(default = None, description="Structured summary dictionary mapping dataset properties.")

    # Dyanmic Routing State Flags
    chosen_model_family: Optional[str] = Field(default = None, description = "The structural routing descision made by the AI agent: 'Linear' or 'Tree'.")
    outliers_detected: bool = Field(default = False, description = "Flag indicating if severe visual outlier clustering or high leverage variance was discovered.")

    # Execution Metrics Repository
    regression_metrics: Dict[str, float] = Field(default_factory = dict, description = "Stores OLS baselines statistics liek R2, ADJ-R2 and F-statistic values." )
    classification_benchmarks: Dict[str, Any] = Field(default_factory = dict, description = "Stores Macro F1-scores, deltas, and feature importances from Function 5.")

    # Production Model & Prediction Outputs
    final_model_metrics: Dict[str, Any] = Field(default_factory = dict, description = "Stores the final production model's evaluation scores (R2, RMSE, F1, accuracy).")
    prediction_output_path: Optional[str] = Field(default = None, description = "Local file path where the prediction results CSV was saved.")
    data_sample_context: Optional[str] = Field(default = None, description = "Text snapshot of the dataset (head + describe) for grounding Q&A responses.")
    sandbox_execution_log: Optional[str] = Field(default = None, description = "Captured output from the last sandbox code execution for debugging visibility.")

    # Critic / Validator Agent Outputs
    critic_validation_report: Dict[str, Any] = Field(default_factory = dict, description = "Full structured validation report: train/test gap, overfitting verdict, ROC-AUC, per-class metrics.")
    critic_flags: List[str] = Field(default_factory = list, description = "Severity flags raised by the critic: OVERFITTING_DETECTED, UNDERFITTING_DETECTED, DATA_LEAKAGE_SUSPECTED.")
    critic_plot_paths: Dict[str, str] = Field(default_factory = dict, description = "Mapping of diagnostic plot names to their saved file paths (confusion_matrix, roc_auc, pr_curve, learning_curve).")

    # Engineering & Self-Correction Loops
    execution_history: List[str] = Field(default_factory = list, description = "Ordered log tracing which graph nodes have executed.")
    error_traceback: Optional[str] = Field(default = None, description = "Captures standard terminal code errors to pass back to code self-correction loops.")