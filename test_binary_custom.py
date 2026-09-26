import time
import numpy as np
from lang_dataset import build_language_dataset
from simple_svm import SimpleSVM_SMO, LinearSVM_Primal
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

np.random.seed(42)

def tfidf_char_features(texts_train, texts_test, max_features=150):
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        max_features=max_features,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(texts_train).toarray()
    X_test = vectorizer.transform(texts_test).toarray()
    return X_train, X_test

def quick_svm_test():
    print("=" * 72)
    print("QUICK SVM TEST - VI vs H'MONG USING lang_dataset")
    print("=" * 72)
    dataset = build_language_dataset(n_per_class=120)
    texts = dataset["text"].tolist()
    labels = np.where(dataset["language"] == "vi", 1.0, -1.0)
    texts_train, texts_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels,
    )
    X_train, X_test = tfidf_char_features(texts_train, texts_test, max_features=150)
    
    t0 = time.time()
    svm_dual = SimpleSVM_SMO(C=1.0, kernel="linear", max_passes=8)
    svm_dual.fit(X_train, y_train)
    t_dual = time.time() - t0
    
    t0 = time.time()
    svm_primal = LinearSVM_Primal(C=1.0, lr=0.01, n_iters=1500)
    svm_primal.fit(X_train, y_train)
    t_primal = time.time() - t0
    
    models = [("Custom Dual/SMO", svm_dual), ("Custom Primal/GD", svm_primal)]
    
    print("\nAccuracy on test set")
    print("-" * 72)
    for name, model in models:
        pred = model.predict(X_test)
        acc = np.mean(pred == y_test)
        print(f"{name:20s} : {acc:.4f}")
        
    n_sv = len(svm_dual.support_)
    print("\nSupport vector check")
    print(f"Support vectors: {n_sv}/{len(y_train)} ({n_sv / len(y_train):.2%})")
    
    w_dual = svm_dual.get_w()
    w_primal = svm_primal.w
    cos = np.dot(w_dual, w_primal) / (np.linalg.norm(w_dual) * np.linalg.norm(w_primal))
    print(f"Dual vs Primal direction similarity: {cos:.4f}")
    
    print("\nTiming")
    print("-" * 72)
    timing = [("Custom Dual/SMO", t_dual), ("Custom Primal/GD", t_primal)]
    for name, t in timing:
        print(f"{name:20s} : {t:7.3f}s")
        
    accs = [np.mean(model.predict(X_test) == y_test) for _, model in models]
    ok = all(a > 0.9 for a in accs)
    print("\nQuick verdict")
    print("PASS" if ok else "FAIL")

if __name__ == "__main__":
    quick_svm_test()
