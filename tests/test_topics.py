from src.topics import fit_lda_tfidf


def test_fit_shapes():
    docs = [
        'Great class, learned a lot about SQL and Python.',
        'Too fast, but the projects were helpful.',
        'Loved the practical examples in Power BI.'
    ]
    vec, lda, X = fit_lda_tfidf(docs, n_topics=2, max_features=50)
    assert X.shape[0] == len(docs)
    assert lda.components_.shape[0] == 2
