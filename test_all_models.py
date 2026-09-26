import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from lang_dataset import build_language_dataset_4_langs
from simple_svm import SimpleSVM_SMO, LinearSVM_Primal

class CustomOVR:
    def __init__(self, estimator_cls, **kwargs):
        self.estimator_cls = estimator_cls
        self.kwargs = kwargs
        self.models = {}
        self.classes_ = []

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for c in self.classes_:
            y_binary = np.where(y == c, 1.0, -1.0)
            model = self.estimator_cls(**self.kwargs)
            model.fit(X, y_binary)
            self.models[c] = model
        return self

    def predict(self, X):
        scores = {}
        for c in self.classes_:
            scores[c] = self.models[c].decision_function(X)
        scores_matrix = np.column_stack([scores[c] for c in self.classes_])
        best_idx = np.argmax(scores_matrix, axis=1)
        return np.array([self.classes_[i] for i in best_idx])

def run_experiment():
    print("=" * 80)
    print("SO SÁNH MÔ HÌNH PHÂN LOẠI NGÔN NGỮ (VI, HMN, EDE, THA) VỚI ONE-VS-REST")
    print("=" * 80)
    df = build_language_dataset_4_langs(n_per_class=300)
    texts = df['text'].values
    labels = df['language'].values
    print(f"Tổng số mẫu: {len(texts)}")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)
    tfidf = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=500, sublinear_tf=True)
    X_train = tfidf.fit_transform(X_train_raw).toarray()
    X_test = tfidf.transform(X_test_raw).toarray()
    
    models = {
        "Custom SVM (SMO)": CustomOVR(SimpleSVM_SMO, C=1.0, kernel="linear", max_passes=5),
        "Custom SVM (Primal)": CustomOVR(LinearSVM_Primal, C=1.0, lr=0.01, n_iters=1000),
        "Sklearn SVC (OvR)": OneVsRestClassifier(SVC(kernel="linear", C=1.0)),
        "Sklearn LinearSVC (OvR)": OneVsRestClassifier(LinearSVC(C=1.0, max_iter=2000, random_state=42)),
        "Perceptron (OvR)": OneVsRestClassifier(Perceptron(max_iter=1000, random_state=42)),
        "Logistic Regression (OvR)": OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=42)),
        "Naive Bayes (OvR)": OneVsRestClassifier(MultinomialNB()),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }

    print("-" * 80)
    print(f"{'Mô hình':<28} | {'Thời gian (s)':<15} | {'Độ chính xác (Acc)':<18}")
    print("-" * 80)
    results = {}
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        t_fit = time.time() - t0
        t0 = time.time()
        preds = model.predict(X_test)
        t_pred = time.time() - t0
        acc = accuracy_score(y_test, preds)
        print(f"{name:<28} | {t_fit + t_pred:<15.4f} | {acc:.4f}")
        results[name] = {"preds": preds}

    print("-" * 80)
    print("\n[Chi tiết] Báo cáo phân loại của Logistic Regression (OvR):")
    best_model_preds = results["Logistic Regression (OvR)"]["preds"]
    print(classification_report(y_test, best_model_preds))

if __name__ == "__main__":
    run_experiment()
