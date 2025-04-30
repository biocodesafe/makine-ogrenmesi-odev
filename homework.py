# Gerekli kütüphaneler
import pandas as pd
import numpy as np

# Veri setini okuyoruz
df = pd.read_excel("Dry_Bean_Dataset.xlsx")

print(df.shape)
print(df.head())

# Random sabitleme
np.random.seed(42)

# Eksik veri ekliyoruz
for col in ['Area', 'Perimeter']:
    df.loc[df.sample(frac=0.05).index, col] = np.nan

df.loc[df.sample(frac=0.35).index, 'MinorAxisLength'] = np.nan

print(df.isnull().sum())

# Eksik verileri dolduruyoruz
df['Area'] = df['Area'].fillna(df['Area'].mean())
df['Perimeter'] = df['Perimeter'].fillna(df['Perimeter'].mean())

# MinorAxisLength sütünunu siliyoruz
df.drop(columns=['MinorAxisLength'], inplace=True)

print(df.isnull().sum())

# Aykırı değer tespiti fonksiyonu
def detect_outliers_iqr(data, feature):
    Q1 = data[feature].quantile(0.25)
    Q3 = data[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[feature] < lower_bound) | (data[feature] > upper_bound)]
    return outliers

# Sayısal sütunlar
types = df.select_dtypes(include=[np.number]).columns.tolist()

for feature in types:
    outliers = detect_outliers_iqr(df, feature)
    print(f"{feature} sütünunda {len(outliers)} tane aykırı değer var.")

# Aykırı değerleri baskılama fonksiyonu
def cap_outliers_iqr(data, feature):
    Q1 = data[feature].quantile(0.25)
    Q3 = data[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data[feature] = np.where(data[feature] < lower_bound, lower_bound, data[feature])
    data[feature] = np.where(data[feature] > upper_bound, upper_bound, data[feature])

for feature in types:
    cap_outliers_iqr(df, feature)

print("Aykırı değerler düzenlendi.")

# StandardScaler kullanarak ölçekleme yapıyoruz
from sklearn.preprocessing import StandardScaler, LabelEncoder

scaler = StandardScaler()

X = df.drop('Class', axis=1)
y = df['Class']

le = LabelEncoder()
y = le.fit_transform(y)

X_scaled = scaler.fit_transform(X)

# PCA için kütüphane
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

print(pca.explained_variance_ratio_)

mean_var = pca.explained_variance_ratio_.mean()
print(f"Ortalama varyans: {mean_var:.4f}")

selected = np.sum(pca.explained_variance_ratio_ > mean_var)
print(f"Seçilen bileşen sayısı: {selected}")

pca_final = PCA(n_components=selected)
X_pca_final = pca_final.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='viridis', s=10)
plt.xlabel('PCA 1. Bileşen')
plt.ylabel('PCA 2. Bileşen')
plt.title('PCA Sonrası')
plt.savefig('pca_sonrasi.png')
plt.show()

# LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components=3)
X_lda = lda.fit_transform(X_scaled, y)

print("LDA ile veri boyutu:", X_lda.shape)

plt.figure(figsize=(8,6))
plt.scatter(X_lda[:,0], X_lda[:,1], c=y, cmap='coolwarm', s=10)
plt.xlabel('LDA 1')
plt.ylabel('LDA 2')
plt.title('LDA Sonrası')
plt.savefig('lda_sonrasi.png')
plt.show()

# Modellemeler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb

from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

models = {
    'LogisticRegression': {
        'model': LogisticRegression(max_iter=1000),
        'params': {'C': [0.01, 0.1, 1, 10]}
    },
    'DecisionTree': {
        'model': DecisionTreeClassifier(),
        'params': {'max_depth': [3, 5, 10]}
    },
    'RandomForest': {
        'model': RandomForestClassifier(),
        'params': {'n_estimators': [50, 100], 'max_depth': [5, 10]}
    },
    'XGBoost': {
        'model': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
        'params': {'n_estimators': [50, 100], 'max_depth': [3, 5]}
    },
    'NaiveBayes': {
        'model': GaussianNB(),
        'params': {}
    }
}

datasets = {
    'Ham': X_scaled,
    'PCA': X_pca_final,
    'LDA': X_lda
}

results = []
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for dataset_name, X_set in datasets.items():
    for model_name, mp in models.items():
        outer_scores = []
        for train_idx, test_idx in outer_cv.split(X_set, y):
            X_train, X_test = X_set[train_idx], X_set[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            clf = GridSearchCV(mp['model'], mp['params'], cv=inner_cv, scoring='accuracy')
            clf.fit(X_train, y_train)

            best_model = clf.best_estimator_
            y_pred = best_model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='macro')
            rec = recall_score(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')

            outer_scores.append([acc, prec, rec, f1])

        outer_scores = np.array(outer_scores)
        mean_scores = outer_scores.mean(axis=0)
        results.append({
            'Dataset': dataset_name,
            'Model': model_name,
            'Accuracy': mean_scores[0],
            'Precision': mean_scores[1],
            'Recall': mean_scores[2],
            'F1': mean_scores[3]
        })

        # ROC için tek bir split ile örnek gösterim
        X_train, X_test, y_train, y_test = train_test_split(X_set, y, test_size=0.2, random_state=42, stratify=y)
        best_model.fit(X_train, y_train)
        y_score = best_model.predict_proba(X_test)

        y_test_bin = label_binarize(y_test, classes=np.unique(y))

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(y_test_bin.shape[1]):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure(figsize=(10,8))
        for i in range(y_test_bin.shape[1]):
            plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC - {model_name} ({dataset_name})')
        plt.legend(loc='lower right')
        plt.savefig(f'roc_{model_name}_{dataset_name}.png')
        plt.close()

results_df = pd.DataFrame(results)
results_df.to_csv('model_degerlendirme.csv', index=False)

print("Tüm sonuçlar ve ROC eğrileri başarıyla kaydedildi.")
