# NLP – Course Feedback Topic Modeling

Discover themes in course feedback using TF‑IDF + LDA topic modeling.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest
```

## Structure
- `src/topics.py` – TF‑IDF + LDA fit/transform utilities
- `notebooks/01_exploration.ipynb` – EDA and modeling steps
- `tests/` – minimal unit tests
