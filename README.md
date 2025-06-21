# NNML Algorithms: Manual Implementation

This repository contains **manual implementations** of classic machine learning algorithms for educational purposes. All core algorithms are coded from scratch **without using libraries like TensorFlow or scikit-learn**. Only basic libraries such as NumPy are used for numerical operations. For some evaluation metrics and plotting, scikit-learn and seaborn/matplotlib are used.

---

## Contents

- **Find-S & Candidate Elimination** (`FindS_Cand.ipynb`): Concept learning algorithms for hypothesis space search.
- **Linear Regression** (`LinearRegression.ipynb`): Manual normalization, error metrics, and regression calculations.
- **Logistic Regression** (`LogisticRegression.ipynb`): Gradient descent, sigmoid, and evaluation—all coded from scratch.
- **Naive Bayes** (`NaiveBias.ipynb`): Gaussian Naive Bayes classifier with manual probability and likelihood calculations.
- **K-Nearest Neighbors (KNN)** (`KNN.ipynb`): Manual distance calculation and prediction logic.
- **Support Vector Machine (SVM)** (`svm.ipynb`): Basic SVM using the primal form and manual updates.
- **Principal Component Analysis (PCA)** (`pca.ipynb`): Dimensionality reduction using manual eigen decomposition.
- **Decision Tree** (`DecisionTree.ipynb`): Entropy, information gain, and simple tree logic from scratch.

---

## Key Features

- **No ML Libraries Used for Algorithms:**  
  All learning algorithms are implemented manually for clarity and educational value.
- **Minimal Dependencies:**  
  Only `numpy` is required for core logic.  
  `matplotlib`, `seaborn`, and `sklearn.metrics` are used **only for plotting and accuracy metrics**.
- **Small, Interpretable Dataset:**  
  All notebooks use a simple car dataset:  
  `[Car Age, Mileage, Fuel Type (0=Petrol, 1=Diesel), Price Category (0=Low, 1=High)]`

---

## Example: Find-S & Candidate Elimination

```python
# Find-S Algorithm
def find_s(X, y):
    hypothesis = X[0].copy()
    for i in range(len(X)):
        if y[i] == 1:
            for j in range(len(X[i])):
                if hypothesis[j] != X[i][j]:
                    hypothesis[j] = '?'
    return hypothesis
```

---

## How to Run

1. Open any `.ipynb` file in Jupyter or VS Code.
2. Run the cells to see step-by-step outputs and explanations.
3. No special setup required beyond `numpy` and (optionally) `matplotlib`, `seaborn`, `sklearn` for metrics/plots.

---

## Educational Purpose

This code is intended for learning and demonstration.  
**No black-box ML libraries are used for the core algorithms.**  
You can see and modify every step of the logic.

---

## License

For educational use only.

---

**Author:**  
Akshita  
NNML Lab, 2025

---

**Note:**  
For any questions or suggestions, please open an issue or contact the author.
