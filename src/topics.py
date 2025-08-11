from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer


def fit_lda_tfidf(docs, n_topics=5, max_features=500):
    """Fit TF‑IDF vectorizer and LDA; return (vectorizer, lda, matrix)."""
    vec = TfidfVectorizer(max_features=max_features, stop_words='english')
    X = vec.fit_transform(docs)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=0, learning_method='batch')
    lda.fit(X)
    return vec, lda, X
