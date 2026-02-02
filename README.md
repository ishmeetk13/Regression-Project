# Boston Housing Price Prediction: Regularization & Polynomial Optimization

This project implements a robust machine learning pipeline to predict median house values in Boston. It demonstrates the transition from a high-bias linear baseline to an optimized **Ridge Regression** model using **Polynomial Feature Expansion** and **L2 Regularization**.

## Key Performance Results
* **Best Model:** Ridge Regression (3rd-Degree Polynomial)
* **R² Score:** **0.835** (Explains 83.5% of the price variance)
* **RMSE:** **3.50** (A ~30% reduction in error compared to the linear baseline)
* **Optimal Hyperparameters:** $\alpha = 4.0$ with a 3rd-degree polynomial transformation



## Project Logic: Solving High Bias
The initial Linear Regression baseline suffered from **High Bias**, as it was too rigid to capture the non-linear relationships in the housing data. 

To solve this, I implemented:
1.  **Polynomial Expansion:** Increased model complexity to a 3rd-degree polynomial to allow the model to fit non-linear trends.
2.  **Ridge (L2) Regularization:** Introduced a penalty on the squared magnitude of coefficients. This was critical to stabilize the 3rd-degree model, preventing the weights from exploding and ensuring the model generalized well to unseen data.



## Methodology
* **Automated Pipelines:** Built using Scikit-Learn `Pipeline` to chain `StandardScaler`, `PolynomialFeatures`, and the estimator.
* **Hyperparameter Tuning:** Conducted an exhaustive search via `GridSearchCV` to find the optimal combination of polynomial degree and alpha penalty.
* **Feature Scaling:** Applied Z-score normalization via `StandardScaler` to ensure that the regularization penalty was applied uniformly across all features regardless of their original units.

## Residual Analysis
The residual plot shows a random scatter around the zero line, confirming that the model has captured the structural patterns in the data. 

---

**Tech Stack:** Python, Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn  
**Open in Colab:** https://colab.research.google.com/#fileId=https%3A//storage.googleapis.com/kaggle-colab-exported-notebooks/ishmeetk13/final-project-supervised-learning-regression.c45ce0ac-4942-42a5-9023-eeabdbbaf981.ipynb%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com/20260202/auto/storage/goog4_request%26X-Goog-Date%3D20260202T103617Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D355638294f10996c1a78aff2099af4ffe05613aa17975ab542893dc5267c9279c8cba0d898d0b04f49cb5503c646a59d249bc1021b8952bcc90d0d824012a26e7420c8f73808e66fe01f0426e04d40de8b7e0248f7aa8c79301ab88d1d4e76648209be03bf453f3ed35e875ac01221105646fbf98e6065167c392cae732749e774fc9f3263c26b7600f3955b3e8b37d4d1c2fa92fddccba8482e2b8a3c1b7c3983e6765274ef6b56e946da7fee7b06aeea50288a1335dc942c0ced6fd93d7a6eb7899ce94bfdde6ad61e970b0aeee0dd63da3b8b8f50d3e71191807863730978f72d4afbd8d683d8a3704bf7da1a4809ce4dbccfe8d4b2deca9273f5b99ece72

**Open in Kaggle:** https://www.kaggle.com/code/ishmeetk13/final-project-supervised-learning-regression

**Dataset:** https://www.kaggle.com/datasets/ishmeetk13/boston-housing-clean-dataset
