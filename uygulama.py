import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# veriyi yüklüyoruz
df = pd.read_csv("movies.csv")

df.info()

missing_value = df.isnull()
print("Eksik değer olan hücreler:\n", missing_value)

# sadece etiketlediğimiz verileri alıyoruz
df_labeled = df[df["mod"].notnull()].copy()
print(f"Etiketli film sayısı: {len(df_labeled)}")

# 3. Tek örnekli sınıfları kaldır
class_counts = df_labeled["mod"].value_counts()
valid_classes = class_counts[class_counts >= 2].index
df_labeled = df_labeled[df_labeled["mod"].isin(valid_classes)]

# 4. Metin sütunlarını birleştir
text_cols = ["overview", "genres", "tagline", "keywords"]
for col in text_cols:
    df_labeled[col] = df_labeled[col].fillna("")

df_labeled["text"] = (
    df_labeled["overview"] + " " +
    df_labeled["genres"] + " " +
    df_labeled["tagline"] + " " +
    df_labeled["keywords"]
)

# 5. Özellikler ve etiket
X = df_labeled["text"]
y = df_labeled["mod"]

# 6. Eğitim / test ayrımı
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 7. Modeller ve pipeline'lar
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVC": LinearSVC(class_weight="balanced")
}

results = {}

# 8. Model karşılaştırması
for name, model in models.items():
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_df=0.9)),
        ("clf", model)
    ])
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
    results[name] = scores
    print(f"{name} CV Accuracy: {np.mean(scores):.4f}")

# 9. En iyi model: Linear SVC ile GridSearch
pipeline_svc = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", LinearSVC(class_weight="balanced"))
])

param_grid = {
    "tfidf__max_df": [0.7, 0.85, 1.0],
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "clf__C": [0.1, 1, 10]
}

grid = GridSearchCV(pipeline_svc, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)

print("\nBest parameters:", grid.best_params_)
print(f"Best CV accuracy: {grid.best_score_:.4f}")

# 10. Değerlendirme
y_pred = grid.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))

# 11. Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=grid.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid.classes_)
disp.plot(xticks_rotation=45)
plt.title("Confusion Matrix - LinearSVC")
plt.tight_layout()
plt.show()

# 12. CV sonuçlarını görselleştir
plt.figure(figsize=(8, 5))
sns.boxplot(data=pd.DataFrame(results))
plt.title("Model CV Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()