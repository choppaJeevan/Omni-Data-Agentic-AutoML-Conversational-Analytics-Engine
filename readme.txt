I divided the architecture into 5 parts

part 1 The Engine room:

We need to build five deterministic functions that our agents will use as Tools to inspect the data and run the baseline statistics

Function 1: The metadata extractor
1. what to build: A function that will take .csv file and load into pandas and extracts the shape and columns, data types and count of missing values, and the cardinality (number of unique values) for categorical columns.
2. why we are build it this way: if you pass a 100MB CSV file directly to alocal LLM, you will instantly run out of VRAM and crash te machine. By converting the dataset into a tiny JSON summary, the LLM can understand the data structure using almost zero memory.
3. Target inference & encoding strategy: by including the column names and unique value counts in summary, the downstream LLM has all the semantic and statistical clues it needs to guess the target variable and decide on encoding paths without ever looking at the raw rows.

Function 2: The Automated preprocessor
1. What to build: A function that takes the dataset and identified target column, splits them into features(X) and target(y), determines the machine learning task type, and executes a mandatory, structural baseline cleanup pipeline:
    a. Dynamic Task Detection: Identifies the target data type. If the target is an object/category or contains < 10 discrete values, it flags the system as classification; otherwise, it flags it as regression.
    b. Target Processing: If the problem is classified as classification, it maps text labels to sequential integers (0, 1, 2...) via Label Encoding instead of One-Hot Encoding to avoid target matrix splitting.
    b. Missing Values: Imputes missing numerical feature columns using the median and missing categorical feature columns using the mode.
    c. Low Cardinality Categories (< 10 unique values): Applies One-Hot Encoding to keep features explicit and highly interpretable.
    d. High Cardinality Categories (> 10 unique values): Applies Target Encoding (mapping categories to the mean of the target variable) or Binary Encoding to completely prevent the Curse of Dimensionality.
2. Why we build it this way: Traditional models crash on missing values or strings. However, blindly using one-hot encoding triggers the Curse of Dimensionality, which tanks model performance and slows down local execution. This adaptive approach ensures the data stays compact, mathematically valid, and highly optimized for your local machine.

Function 3: The Baseline OLS & Residual plotter
1. What to build: A function that takes the globally preprocessed $X$ and $y$ (if the task is flagged as regression), fits a baseline Ordinary Least Squares (OLS) linear regression using statsmodels, extracts core performance metrics (R^2, Adjusted R^2, F-statistic, p-values), and exports a visual diagnostic residual plot image (residual_plot.png) to a local folder.
2. Why we build it this way: We utilize statsmodels instead of scikit-learn for our regression baseline because statsmodels generates text-dense, highly descriptive statistical tables that local text LLMs excel at parsing for coefficients. We save the plot locally so our local Vision LLM can programmatically inspect its geometric structure in Part 4.

Function 4: The Model-Specific Optimizer (Conditional Tool)
1. What to build: A specialized precision function containing 2 explicit data surgery options triggered dynamically by the agent's algorithmic path:
    a. Outlier Strategy: Uses a statistical method like the Interquartile Range (IQR) threshold or a fast IsolationForest to identify and clip extreme leverage data points that warp linear boundaries.
    b. Feature scaling: Applies a RobustScaler (scaling via the median and Interquartile Range) instead of standard Z-score normalization (StandardScaler) to bring all numerical features onto the same mathematical scale.
2. Why we build it this way: 
    a. Preserving Feature Variance: Using RobustScaler prevents extreme outliers from skewing the mean and standard deviation. It scales data based on its center mass (the middle 50%), ensuring that normal data points don't get squashed together, which would destroy the geometric boundaries that distance-based models rely on.
    b. Just-In-Time (JIT) Execution: If the agent analyzes the baseline residual plot and says, "Let's train an SVM or regularized Logistic Regression," it passes the data through this function first. If the agent concludes, "The baseline residuals look highly non-linear, let's skip straight to a Random Forest," it completely bypasses this tool.

Function 5: The Classifer Benchmark
1. What to build: A benchmarking engine triggered dynamically if the problem type is flagged as classification that splits data, corrects for class skew, trains two distinct estimators simultaneously on a local validation split, and extracts performance metrics:
    a. The Linear Baseline: Trains a Logistic Regression model with L2 regularization to construct flat, straight decision hyperplanes.
    b. The Non-Linear Challenger: Trains a Random Forest Classifier or HistGradientBoostingClassifier to draw complex, step-like, non-linear boundaries.
    c. Imbalance Protection Heuristics: Enforces Stratified Splitting during the data separation phase to guarantee class ratio preservation. Injects the class_weight='balanced' penalty parameter into both estimators to dynamically scale loss penalties inversely to class frequencies.
    d. Metrics Extraction: Evaluates both models using the Macro F1-Score instead of basic Accuracy. It logs the F1-Score delta (Delta) between models and outputs a sorted dictionary of feature importances from the tree model.
2. Why we build it this way: Categorical targets cannot be ranked numerically or parsed via OLS residual plots. This function replaces visual inspection with a performance matrix benchmark. If the target classes are severely imbalanced (e.g., 95% vs 5%), standard accuracy scores create a "lazy predictor" trap where a broken model looks perfect on paper. Evaluating via Macro F1 with balanced class weights forces the models to prioritize minority classes equally, giving the text LLM a clear mathematical signature to decide if the feature boundaries are linear or non-linear.

Part 2 The graph backbone & Local Orchestration:

We need to build a centralized routing loop and a local multi-modal framework that acts as our agent's nervous system and brain, ensuring data artifacts and routing decisions move safely without cloud dependencies.

Component 1: The pydantic Center Graph State
1. What to build: A single, centralized, type-safe Python data structure that maintains the active memory of the entire system. 
It explicity tracks the file paths (CSV and generated PNGs), detected traget schemas, the current problem type(Regression vs classification), model selection flags, performance metrics, execution history logs and code tracebacks
2. Why we build it this way: Instead of passing massive data arrays directly between isolated functions - which causes severe variable leaking and fragile code dependencies - the agent passes a lightweight state object. Every node in the pipeline reads from this shared context and writes its updates back to it. This design decouples our data engineering logic from our agent routing logic, making the entire pipeline modular, easily debuggable, and perfectly compatible with framework graphs.

Component 2: The local Model router 
1. What to build: A deterministic start router that intercepts the state immediately after the preprocessor completes. If problem_type equals regression, it forces a hard execution branch to OLS baseline tool. If problem_type equals classfication, it automatically reroutes the workflow to the Classifier Benchmark tool.
2. Why we build it this way: Hardcoding this baseline branch deterministically completely removes the risk of an LLM hallucinating its initial data path. We don't waste precious processing cycles asking a text model what to do next when basic statistical data checking already dictates the correct route. This ensures your local architecture runs at peak efficiency.

Component 3: The text Critic Node(Ollama Text Orchestrator)
1. What to build: An abstraction layer that communicates with a fast, highly quantized local open-source LLM (such as Llama 3 8B or Mistral 7B) running via Ollama. This node parses text data arrays - specifically the JSON schema summary from Function 1 or the classification F1-Score delta from Function 5 - to make executive structural decisions (e.g., selecting between linear and tree models or naming the correct target feature).
2. Why we build it this way: Running massive proprietary cloud models is complete overkill for parsing clean JSON text arrays or comparing two numerical scores. A small, local 7B or 8B model fits entirely inside your system's VRAM, operates with near-zero latency, ensures 100% data privacy on your machine, and excels at structured downstream text extraction when given explicit system prompts.

Component 4: The Visual Judge Node (Ollama Multimodal Orchestrator)
1. What to build: A specialized multimodal analysis node that targets a local vision-language model (like LLaVA 7B) through Ollama. This node is triggered exclusively on the regression path; it programmatically loads the locally saved residual_plot.png, inspects the geometric distribution of the scatter plot points, and explicitly outputs whether it detects heteroscedasticity funnels, non-linear curvature, or compressed outlier clusters.
2. Why we build it this way: Text-only LLMs are completely blind to visual data; they cannot "see" a graph. If you rely purely on mathematical metrics like $R^2$, you can easily miss severe underlying data violations (like non-linear patterns hidden behind a deceptively decent score). By embedding a dedicated vision model to read the actual diagnostic image, your agent mimics a meticulous human data scientist - using its "eyes" to spot visual mathematical anomalies before deploying an architecture.