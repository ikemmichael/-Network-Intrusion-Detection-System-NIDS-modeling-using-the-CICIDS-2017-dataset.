# ======================================================
# CICIDS 2017 Machine Learning Pipeline (FINAL + OPTIMIZED)
# Kali Linux + Pandas 3.x Compatible
# Fast + Stable + Report Ready
# ======================================================


# ======================================================
# 1. IMPORTS
# ======================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

# FAST models
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier



# ======================================================
# 2. LOAD DATASET
# ======================================================
print("\n[+] Loading dataset...")

df = pd.read_csv("CICI.csv", low_memory=False)

# Remove hidden spaces (VERY important for CICIDS)
df.columns = df.columns.str.strip()

print(df.head())
print("Dataset shape:", df.shape)



# ======================================================
# 3. OPTIONAL SAMPLING (UNCOMMENT FOR LAPTOPS)
# ======================================================
# df = df.sample(50000, random_state=42)



# ======================================================
# 4. CLEAN DATA
# ======================================================
print("\n[+] Cleaning dataset...")

# Replace infinite values
df.replace([np.inf, -np.inf], np.nan, inplace=True)

print("Missing values per column:\n", df.isnull().sum())

# Fill numeric NaN with mean
df.fillna(df.mean(numeric_only=True), inplace=True)

# Drop any remaining
df.dropna(inplace=True)

print("After cleaning shape:", df.shape)



# ======================================================
# 5. LABEL ENCODING
# ======================================================
print("\n[+] Encoding labels...")

le = LabelEncoder()
df["Label"] = le.fit_transform(df["Label"])



# ======================================================
# 6. FEATURES + SCALING (memory optimized)
# ======================================================
print("\n[+] Scaling features...")

X = df.drop("Label", axis=1)
y = df["Label"]

scaler = StandardScaler()

# .values is slightly faster + lower RAM
X_scaled = scaler.fit_transform(X.values)

print("Instances:", X.shape[0])
print("Features:", X.shape[1])



# ======================================================
# 7. DATA EXPLORATION PLOT
# ======================================================
plt.figure(figsize=(7,4))
sns.countplot(x=y)
plt.title("Attack vs Normal Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()



# ======================================================
# 8. TRAIN / TEST SPLIT (STRATIFIED ✔)
# ======================================================
print("\n[+] Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



# ======================================================
# 9. MODEL EVALUATION FUNCTION (IMPROVED)
# ======================================================
def evaluate_model(model, name):
    print(f"\n[+] Training {name}...")

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    print(f"\n{name} Results")
    print("Accuracy:", acc)
    print(classification_report(y_test, preds))

    # -----------------------
    # Confusion Matrix
    # -----------------------
    ConfusionMatrixDisplay.from_predictions(y_test, preds)
    plt.title(f"{name} Confusion Matrix")
    plt.show()

    # -----------------------
    # ROC Curve (binary only)
    # -----------------------
    try:
        scores = model.decision_function(X_test)
    except:
        scores = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{name} ROC Curve")
    plt.legend()
    plt.show()

    # -----------------------
    # Cross Validation
    # -----------------------
    print("[+] Cross-validating...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=3, n_jobs=-1)
    print("CV Accuracy:", cv_scores.mean())

    return acc



# ======================================================
# 10. TRAIN MODELS
# ======================================================
svm_acc = evaluate_model(
    LinearSVC(max_iter=5000),
    "Linear SVM"
)

dt_acc = evaluate_model(
    DecisionTreeClassifier(),
    "Decision Tree"
)

rf = RandomForestClassifier(
    n_estimators=100,
    n_jobs=-1
)

rf_acc = evaluate_model(
    rf,
    "Random Forest"
)



# ======================================================
# 11. MODEL COMPARISON
# ======================================================
print("\n[+] Comparing models...")

results = {
    "Linear SVM": svm_acc,
    "Decision Tree": dt_acc,
    "Random Forest": rf_acc
}

pd.DataFrame(results.items(), columns=["Model", "Accuracy"]).plot(
    x="Model", y="Accuracy", kind="bar"
)

plt.title("Model Comparison")
plt.ylabel("Accuracy")
plt.show()



# ======================================================
# 12. FEATURE IMPORTANCE (Random Forest)
# ======================================================
print("\n[+] Feature Importance...")

importances = rf.feature_importances_
feat_imp = pd.Series(importances, index=X.columns)

feat_imp.nlargest(10).plot(kind="barh")
plt.title("Top 10 Important Features")
plt.show()



# ======================================================
# DONE
# ======================================================
print("\n[✓] Pipeline completed successfully!")
