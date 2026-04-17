Rocket Loans: Advanced Loan Approval Prediction

This project implements a comprehensive machine learning pipeline to predict loan eligibility based on applicant profiles. It transitions from traditional statistical analysis and exploratory data research (EDA) to advanced ensemble modeling and deep learning architectures.

🚀 Key Features
Advanced Imputation: Utilized IterativeImputer and KNNImputer to handle missing values in critical features like Credit_Score and Amount_Disbursed.
Feature Engineering: Engineered "Total Income" metrics and performed binning for frequency distribution analysis.
Hybrid Modeling: Comparison of traditional classifiers (Logistic Regression, SVM) vs. Ensembles (XGBoost, Random Forest, AdaBoost) vs. Deep Learning (TensorFlow/Keras).
Dimensionality Reduction: Implemented PCA to analyze feature variance and RFE (Recursive Feature Elimination) for optimal feature selection.
Class Imbalance Handling: Addressed via Stratified K-Fold cross-validation to ensure model reliability across the minority class (rejected loans).

🛠️ Tech Stack
Languages: Python 3.x
Data Science: Pandas, NumPy, SciPy (Statistical Analysis)
Machine Learning: Scikit-Learn (PCA, Iterative Imputer, RandomizedSearchCV, XGBoost)
Deep Learning: TensorFlow / Keras (Sequential API)
Visualization: Matplotlib, Seaborn

📊 Dataset Analysis
The dataset includes applicant demographics and financial history. Key findings during EDA:
Skewness: Identified highly positively skewed distributions in Loan_Bearer_Income and Amount_Disbursed.
Outlier Detection: Automated custom summary functions were used to detect and quantify outliers in the financial features.
Correlation: Analyzed via VIF (Variance Inflation Factor) to ensure no multi-collinearity between feature sets.

📈 Model Performance
The pipeline evaluates multiple models using F1-Score and Accuracy as primary metrics:
Baseline: Logistic Regression
Ensembles: XGBoost and Gradient Boosting (Optimized via Hyperparameter Tuning)
Neural Network: A Multi-Layer Perceptron (MLP) built with Keras for capturing non-linear relationships.
📈 Model Performance Benchmarking
I conducted a comprehensive comparison using 10-fold Stratified Cross-Validation. The models were evaluated on Accuracy and F1-Score to account for class imbalance.
Model Name	Accuracy	F1 Score
Decision Tree Classifier	83.06%	0.888
AdaBoost Classifier	83.06%	0.888
Logistic Regression	82.90%	0.887
Gradient Boosting	82.73%	0.886
XGBoost Classifier	81.11%	0.872
Key Insight: While XGBoost is often a go-to, the Decision Tree and AdaBoost models achieved the highest F1-Score (0.888), making them the most reliable choices for minimizing false approvals.

💻 How to Run
Clone the repository:
git clone [(https://github.com/jessyvijayan/loan_approval_prediction.git)]

Install the dependencies:
Using the requirements.txt file ensures all library versions match the development environment:
pip install -r requirements.txt

Launch the Application:
To run the interactive Streamlit dashboard:
streamlit run [streamlit_loan_approval.py](https://github.com/jessyvijayan/loan_approval_prediction/blob/main/streamlit_loan_approval.py)

