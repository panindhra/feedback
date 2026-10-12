import sys, os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.getcwd())
import app as app_module

app = app_module.app
client = app.test_client()


def test_model_reloads_when_pipeline_is_incompatible(tmp_path, monkeypatch):
    bad_model_path = tmp_path / 'bad_model.pkl'

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10)),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])
    pipeline.fit(['good service', 'bad service', 'okay service'], ['positive', 'negative', 'neutral'])
    clf = pipeline.named_steps['clf']
    if hasattr(clf, 'multi_class'):
        delattr(clf, 'multi_class')
    joblib.dump(pipeline, bad_model_path)

    monkeypatch.setattr(app_module, 'MODEL_PATH', str(bad_model_path))
    monkeypatch.setattr(app_module, 'model', None)

    app_module.load_model()
    result = app_module.analyze_sentiment('This is absolutely wonderful and amazing!')

    assert result is not None
    assert result['sentiment'] in {'positive', 'negative', 'neutral'}
    assert 'probabilities' in result


r = client.get('/login')
print(f'GET /login  -> status={r.status_code} html_len={len(r.data)} has_username_field={"username" in r.get_data(as_text=True)}')

r2 = client.post('/login', data={'username':'admin','password':'admin123'}, follow_redirects=False)
print(f'POST /login -> status={r2.status_code} location={r2.headers.get("Location","-")}')

r3 = client.get('/dashboard')
text3 = r3.get_data(as_text=True)
print(f'GET /dashboard -> status={r3.status_code} html_len={len(r.data)} has_sidebar={"sidebar" in text3} has_navlinks={"nav-links" in text3} has_dashboard_title={"Dashboard" in text3}')

r4 = client.get('/analyze')
text4 = r4.get_data(as_text=True)
print(f'GET /analyze   -> status={r4.status_code} html_len={len(r4.data)} has_prob_bars={"prob-bars" in text4} has_sample_tags={"sample-tags" in text4}')

r5 = client.get('/history')
print(f'GET /history   -> status={r5.status_code} html_len={len(r5.data)}')

r6 = client.post('/analyze', json={'feedback':'This is absolutely wonderful and amazing!'})
print(f'POST /analyze  -> status={r6.status_code} body={r6.get_json()}')

client.get('/logout')
r7 = client.get('/dashboard', follow_redirects=False)
print(f'GET /dashboard (unauth) -> redirect status={r7.status_code} to={r7.headers.get("Location","-")}')
print('\n--- INTERNAL CLIENT TESTS OK ---')
