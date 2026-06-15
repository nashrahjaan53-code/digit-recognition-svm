import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, os

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

os.makedirs("../model", exist_ok=True)
print("✅ All libraries imported!")

digits = load_digits()
X, y = digits.data, digits.target
print(f"✅ Dataset: {X.shape[0]} samples | {X.shape[1]} features | Classes: {np.unique(y)}")


plt.figure(figsize=(10,4))
sns.countplot(x=y, palette='tab10')
plt.title('Class Distribution of Digits (0–9)', fontsize=14, fontweight='bold')
plt.xlabel('Digit'); plt.ylabel('Count')
plt.tight_layout()
plt.savefig('../model/class_distribution.png', dpi=150)
plt.close()

fig, axes = plt.subplots(2, 10, figsize=(15, 4))
fig.suptitle('Sample Images — One per Digit (0–9)', fontsize=14, fontweight='bold')
for digit in range(10):
    idxs = np.where(y == digit)[0]
    for row, i in enumerate([0, 5]):
        axes[row, digit].imshow(X[idxs[i]].reshape(8,8), cmap='gray_r')
        if row == 0: axes[row, digit].set_title(f'{digit}', fontsize=11, fontweight='bold')
        axes[row, digit].axis('off')
plt.tight_layout()
plt.savefig('../model/sample_images.png', dpi=150)
plt.close()
print("✅ EDA plots saved.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_sc)
X_test_pca  = pca.transform(X_test_sc)
print(f"✅ PCA: kept {pca.n_components_} components (95% variance)")

plt.figure(figsize=(9,4))
cumvar = np.cumsum(pca.explained_variance_ratio_)*100
plt.plot(cumvar, color='steelblue', linewidth=2)
plt.axhline(95, color='red', linestyle='--', label='95% threshold')
plt.xlabel('Number of Components'); plt.ylabel('Cumulative Variance (%)')
plt.title('PCA — Cumulative Explained Variance', fontsize=13, fontweight='bold')
plt.legend(); plt.tight_layout()
plt.savefig('../model/pca_variance.png', dpi=150)
plt.close()


print("\n Running GridSearchCV (this may take ~1–2 min)...")
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto'], 'kernel': ['rbf']}
grid = GridSearchCV(SVC(probability=True, random_state=42), param_grid,
                    cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid.fit(X_train_pca, y_train)
best = grid.best_estimator_
print(f"\n✅ Best Params  : {grid.best_params_}")
print(f"✅ Best CV Acc  : {grid.best_score_*100:.2f}%")


y_pred = best.predict(X_test_pca)
acc = (y_pred == y_test).mean()
print(f"\n📊 Test Accuracy: {acc*100:.2f}%")
print("\n📋 Classification Report:\n")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(9,7))
ConfusionMatrixDisplay(cm, display_labels=range(10)).plot(ax=ax, cmap='Blues', colorbar=True)
ax.set_title('Confusion Matrix — SVM Digit Recognition', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('../model/confusion_matrix.png', dpi=150)
plt.close()

wrong = np.where(y_pred != y_test)[0]
fig, axes = plt.subplots(2, 5, figsize=(12,5))
fig.suptitle('❌ Misclassified Examples', fontsize=13, fontweight='bold')
for i, idx in enumerate(wrong[:10]):
    ax = axes[i//5, i%5]
    img = pca.inverse_transform(X_test_pca[idx].reshape(1,-1))
    img = scaler.inverse_transform(img.reshape(1,-1)).reshape(8,8)
    ax.imshow(img, cmap='gray_r')
    ax.set_title(f'True:{y_test[idx]} Pred:{y_pred[idx]}', color='red', fontsize=9)
    ax.axis('off')
plt.tight_layout()
plt.savefig('../model/misclassified.png', dpi=150)
plt.close()
print("✅ All evaluation plots saved.")


results = pd.DataFrame(grid.cv_results_)
pivot = results.pivot_table(values='mean_test_score',
                             index='param_C', columns='param_gamma')
plt.figure(figsize=(7,5))
sns.heatmap(pivot*100, annot=True, fmt='.2f', cmap='YlGnBu')
plt.title('GridSearchCV — Accuracy Heatmap (C vs Gamma)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('../model/gridsearch_heatmap.png', dpi=150)
plt.close()
print("✅ GridSearch heatmap saved.")

joblib.dump(best,   '../model/svm_model.pkl')
joblib.dump(scaler, '../model/scaler.pkl')
joblib.dump(pca,    '../model/pca.pkl')
print("\n Model saved!")
print(f"\n DONE! Test Accuracy: {acc*100:.2f}% | Best: {grid.best_params_}")
