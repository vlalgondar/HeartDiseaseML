# Heart Disease Prediction using Machine Learning

## Overview
Heart disease remains a leading cause of death globally, emphasizing the importance of early detection and intervention. This project aims to predict the presence of heart disease using machine learning algorithms trained on the UCI Heart Disease dataset. Our approach includes comprehensive preprocessing, model selection, and evaluation to identify the best performing algorithm for this classification task.

## Key Features
- **Dataset**: UCI Heart Disease dataset containing 303 records and 14 attributes related to heart health.
- **Algorithms**: Evaluated Support Vector Machine (SVM), Decision Tree, Random Forest, and K-Nearest Neighbors (KNN).
- **Best Model**: SVM achieved a 90% accuracy on the final test set.
- **Techniques**:
  - Feature selection and scaling using `StandardScaler`.
  - Hyperparameter tuning with `GridSearchCV`.
  - Model validation using 5-fold cross-validation.
  - Final evaluation with a confusion matrix and detailed metrics (precision, recall, F1-score).

## Methodology
1. **Data Preprocessing**:
   - Selected relevant features to train models.
   - Split the data into training (80%) and testing (20%) sets.
   - Normalized features using `StandardScaler`.

2. **Model Training**:
   - Implemented and evaluated four models: SVM, Decision Tree, Random Forest, and KNN.
   - Tuned hyperparameters using grid search for optimal performance.
   - Used 5-fold cross-validation for model validation.

3. **Model Evaluation**:
   - Assessed performance using accuracy, precision, recall, and F1-score.
   - Selected SVM for its balanced precision-recall trade-off and robustness.

4. **Final Validation**:
   - Conducted validation on a withheld test set to simulate real-world application.
   - SVM demonstrated 90% accuracy with high precision and recall for both classes.

## Results
- **SVM Performance**:
  - Precision: 89% (no disease), 91% (disease).
  - Recall: 89% (no disease), 91% (disease).
  - F1-Score: 0.89 (no disease), 0.91 (disease).
  - Accuracy: 90%.

- **Comparison with Other Models**:
  - Decision Tree: 77% accuracy.
  - Random Forest: 81% accuracy.
  - KNN: 79% accuracy.

## Lessons Learned
- The importance of data preprocessing and scaling for machine learning performance.
- Effective hyperparameter tuning can significantly improve model outcomes.
- SVM's robustness makes it suitable for medical diagnostics.

## Tools and Skills
- **Programming Language**: Python
- **Libraries**: Scikit-learn, Pandas, NumPy, Matplotlib
- **Machine Learning Techniques**: Supervised learning, hyperparameter tuning, cross-validation
- **Evaluation Metrics**: Accuracy, precision, recall, F1-score, confusion matrix
- **Project Management**: Version control with Git/GitHub

## Potential Applications
This project demonstrates the potential of machine learning for predictive healthcare. Expanding this approach could include larger datasets, diverse demographics, and exploring advanced algorithms to improve diagnostic accuracy.
