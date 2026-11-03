import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.inspection import permutation_importance

os.makedirs("outputs", exist_ok=True)

# 1. Load data (German Credit dataset, downloaded automatically)
data = fetch_openml("credit-g", version=1, as_frame=True)
X = data.data
y = (data.target == "bad").astype(int)  # 1 = bad credit risk
print("Shape:", X.shape)
print("Class balance:\n", y.value_counts(normalize=True))

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3. Preprocessing
cat_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

# 4. Compare 3 models with 5-fold cross-validation
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {}
for name, model in models.items():
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    results[name] = scores.mean()
    print(f"{name}: ROC-AUC = {scores.mean():.3f} (+/- {scores.std():.3f})")

pd.Series(results).sort_values().plot(kind="barh", title="Model comparison (CV ROC-AUC)")
plt.tight_layout()
plt.savefig("outputs/model_comparison.png")
plt.close()

# 5. Hyperparameter tuning on Gradient Boosting
pipe = Pipeline([("prep", preprocessor), ("model", GradientBoostingClassifier(random_state=42))])
param_grid = {
    "model__n_estimators": [100, 200],
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [2, 3],
}
grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
grid.fit(X_train, y_train)
best = grid.best_estimator_
print("Best params:", grid.best_params_)

# 6. Final evaluation on the held-out test set
proba = best.predict_proba(X_test)[:, 1]
preds = (proba >= 0.5).astype(int)
print("Test ROC-AUC:", round(roc_auc_score(y_test, proba), 3))
print(classification_report(y_test, preds, target_names=["good", "bad"]))

ConfusionMatrixDisplay.from_predictions(y_test, preds)
plt.savefig("outputs/confusion_matrix.png")
plt.close()

RocCurveDisplay.from_predictions(y_test, proba)
plt.savefig("outputs/roc_curve.png")
plt.close()

# 7. Explainability: which features matter most?
imp = permutation_importance(best, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=42)
imp_series = pd.Series(imp.importances_mean, index=X_test.columns).sort_values().tail(10)
imp_series.plot(kind="barh", title="Top 10 feature importances (permutation)")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png")
plt.close()

# 8. Save the model
joblib.dump(best, "outputs/credit_model.joblib")
print("Done! Model and plots saved in outputs/")