# Credit Card Fraud Shield

**Intelligent Transaction Security** — A full-stack machine learning web application for detecting suspicious credit card transactions, benchmarking imbalanced classifiers, exploring density anomalies, and auditing prediction history in SQLite.

## Dataset Setup (Kaggle ULB Credit Card Fraud)

1. Download `creditcard.csv` from Kaggle:
   [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Either:
   - Place the file at `data/creditcard.csv` and click **Train Local creditcard.csv** in the UI, or
   - Upload `creditcard.csv` directly on the **Dashboard** or **Settings** page.
3. For quick local testing without downloading the 150 MB Kaggle archive, click **Train ULB Benchmark (6,000 rows)** to generate a schema-compatible synthetic benchmark dataset and run the full ML pipeline. No fabricated model metrics are ever displayed.

### Note on PCA Features (`V1–V28`)
The ULB dataset contains 31 columns (`Time`, `V1`–`V28`, `Amount`, `Class`), where `V1–V28` are already PCA-transformed components from the original transaction attributes to protect cardholder confidentiality. Therefore, repeated PCA is not applied as mandatory preprocessing; instead, an optional **PCA Dimensionality Reduction Experiment** is provided on the **Fraud Analytics** page to compare explained variance and classification performance before and after a secondary 95%-variance projection.

## Machine Learning Pipeline

- **Zero-Leakage Preprocessing (`server/ml/preprocessing.ts`)**: Stratified 70% Train / 15% Validation / 15% Test split. `RobustScaler` (median & IQR) on `Time` and `Amount` plus feature standardization parameters are fitted strictly on the training split.
- **Models (`server/ml/models.ts`)**:
  1. **Logistic Regression**: L2-regularized Adam optimization with inverse-class-frequency positive class weighting.
  2. **Decision Tree**: Class-weighted CART Gini impurity classifier with depth and leaf-size pruning.
  3. **Tuned XGBoost**: Regularized second-order gradient boosted trees with stratified 3-fold cross-validation hyperparameter search (`n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`), validation-set early stopping, and validation-set decision threshold optimization.
- **Unsupervised Experiments (`server/ml/experiments.ts`)**:
  - **PCA Experiment**: 30×30 Jacobi eigendecomposition on standardized training features.
  - **DBSCAN Anomaly Detection**: Density clustering on top principal components to mitigate high-dimensional distance concentration; noise points (`-1`) are evaluated post-hoc against fraud labels strictly as structural anomalies.
- **Persistence (`server/database.ts`)**: Built-in SQLite (`database/transactions.db`) storing predictions, fraud probabilities, risk tiers, and threshold settings with duplicate prevention and zero cardholder PII storage.

## Running Locally

```bash
npm install
npm run dev
```

Open `http://localhost:3000`.
