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
💻 How to Run
Clone the repo: git clone (https://github.com/jessyvijayan/loan_approval_prediction.git)
Install dependencies: pip install -r requirements.txt
Run the notebook or script: python loan_prediction.py
