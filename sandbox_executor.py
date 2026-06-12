import sys
import io
import traceback
import textwrap
from typing import Dict, Any

def execute_sandboxed_code(generated_code: str, shared_context: Dict[str, Any]) -> dict:
    """
    Executes an AI-generated Python script inside an isolated namespace.
    Captures stdout metrics, catches errors for self-correction, and returns state updates.
    """
    print("\n [Sandbox] Allocating isolated runtime memory space...")

    # Redirect standard output so we can capture print statements from the code
    old_stdout = sys.stdout
    redirected_output = sys.stdout = io.StringIO()

    # Establish a local sandboxed namespace containing only necessary variables
    plt_module = shared_context.get("plt")
    if plt_module:
        # Intercept plt.show() to prevent it from freezing the terminal
        plt_module.show = lambda *args, **kwargs: print(" [Sandbox Warning] Blocked plt.show() to prevent terminal freeze. Use savefig.")
        
    sandbox_globals = shared_context.copy()
    sandbox_globals["plt"] = plt_module
    sandbox_globals["metrics_output"] = {}

    execution_success = False
    captured_logs = ""
    error_message = None
    
    try:
        # Run the dynamically generated string as native Python code
        exec(textwrap.dedent(generated_code), sandbox_globals)
        
        # Restore terminal output control and capture logs
        sys.stdout = old_stdout
        captured_logs = redirected_output.getvalue()
        execution_success = True
        print(" [Sandbox] Code executed successfully with zero runtime crashes.")

    except Exception as e:
        # Capture the terminal traceback if the code fails
        sys.stdout = old_stdout
        captured_logs = redirected_output.getvalue()
        
        # Extract full error traceback stack trace so the LLM can read the line breakdown
        error_lines = traceback.format_exception(*sys.exc_info())
        error_message = "".join(error_lines)
        print(f" [Sandbox Runtime Error Flagged]:\n{error_message}")

    return {
        "success": execution_success,
        "logs": captured_logs,
        "error_traceback": error_message,
        "extracted_metrics": sandbox_globals.get("metrics_output", {})
    }

