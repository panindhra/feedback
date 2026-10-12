import os
import uuid
from datetime import datetime
import joblib
from functools import wraps
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from train import preprocess_text

app = Flask(__name__)
app.secret_key = 'feedback-analyzer-secret-key-2026'
app.permanent_session_lifetime = 7 * 24 * 60 * 60

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'sentiment_model.pkl')

model = None

USERS = {
    'admin': {'password': 'admin123', 'name': 'Admin User', 'email': 'admin@example.com'},
    'demo': {'password': 'demo123', 'name': 'Demo User', 'email': 'demo@example.com'},
    'user': {'password': 'user123', 'name': 'John Doe', 'email': 'john@example.com'},
    'syam': {'password': 'syam123', 'name': 'Syam', 'email': 's@example.com'}
    
}


def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            candidate = joblib.load(MODEL_PATH)
            estimator = candidate.steps[-1][1] if hasattr(candidate, 'steps') else candidate
            if not hasattr(estimator, 'predict_proba') or not hasattr(estimator, 'classes_'):
                raise AttributeError('Model artifact is missing required sklearn attributes.')
            if hasattr(estimator, 'multi_class') is False and estimator.__class__.__name__ == 'LogisticRegression':
                raise AttributeError('Model artifact is incompatible with the installed sklearn version.')
            model = candidate
            print("Model loaded successfully!")
            return
        except Exception as exc:
            print(f"Model at {MODEL_PATH} is incompatible or corrupt ({exc}). Retraining...")
            from train import train_model
            model = train_model()
            return
    print(f"Model not found at {MODEL_PATH}. Run train.py first.")


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


def analyze_sentiment(text):
    if model is None:
        load_model()
    if model is None:
        return None

    estimator = model.steps[-1][1] if hasattr(model, 'steps') else model
    if not hasattr(estimator, 'multi_class') or not hasattr(estimator, 'predict_proba'):
        print('Detected incompatible model artifact. Reloading trained model...')
        load_model()
        if model is None:
            return None
        estimator = model.steps[-1][1] if hasattr(model, 'steps') else model

    clean_text = preprocess_text(text)
    prediction = model.predict([clean_text])[0]
    probabilities = model.predict_proba([clean_text])[0]
    classes = model.classes_
    proba_dict = {cls: round(float(prob), 4) for cls, prob in zip(classes, probabilities)}
    return {
        'sentiment': prediction,
        'confidence': round(float(max(probabilities)), 4),
        'probabilities': proba_dict
    }


def get_history():
    if 'history' not in session:
        session['history'] = []
    return session['history']


def add_history(records):
    history = get_history()
    for r in records:
        entry = {
            'id': str(uuid.uuid4())[:8],
            'feedback': r['feedback'],
            'sentiment': r['sentiment'],
            'confidence': r.get('confidence', 0),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M')
        }
        history.insert(0, entry)
    session['history'] = history[:200]
    session.modified = True


def get_dashboard_stats():
    history = get_history()
    total = len(history)
    pos = sum(1 for h in history if h['sentiment'] == 'positive')
    neg = sum(1 for h in history if h['sentiment'] == 'negative')
    neu = sum(1 for h in history if h['sentiment'] == 'neutral')
    if total == 0:
        total = 0
        pos_pct = neg_pct = neu_pct = 0
    else:
        pos_pct = round(pos / total * 100, 1)
        neg_pct = round(neg / total * 100, 1)
        neu_pct = round(neu / total * 100, 1)
    return {
        'total': total,
        'positive': pos, 'negative': neg, 'neutral': neu,
        'positive_pct': pos_pct, 'negative_pct': neg_pct, 'neutral_pct': neu_pct,
        'recent': history[:5]
    }


@app.route('/')
def index():
    if 'user' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip().lower()
        password = request.form.get('password', '')
        user = USERS.get(username)
        if user and user['password'] == password:
            session.permanent = True
            session['user'] = {
                'username': username,
                'name': user['name'],
                'email': user['email']
            }
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html',
                                   error='Invalid username or password. Try admin/admin123 or demo/demo123 or user/user123',
                                   username=username)

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user', None)
    session.pop('history', None)
    return redirect(url_for('login'))


@app.route('/dashboard')
@login_required
def dashboard():
    stats = get_dashboard_stats()
    return render_template('dashboard.html', page='dashboard', stats=stats, user=session['user'])


@app.route('/analyze', methods=['GET'])
@login_required
def analyze_page():
    return render_template('analyze.html', page='analyze', user=session['user'])


@app.route('/analyze', methods=['POST'])
@login_required
def analyze():
    data = request.get_json()
    text = data.get('feedback', '').strip()

    if not text:
        return jsonify({'error': 'Please provide some feedback text.'}), 400

    result = analyze_sentiment(text)
    if result is None:
        return jsonify({'error': 'Model not trained yet. Please run train.py first.'}), 503

    add_history([{'feedback': text, 'sentiment': result['sentiment'], 'confidence': result['confidence']}])

    return jsonify({
        'feedback': text,
        'sentiment': result['sentiment'],
        'confidence': result['confidence'],
        'probabilities': result['probabilities']
    })


@app.route('/analyze_batch', methods=['POST'])
@login_required
def analyze_batch():
    data = request.get_json()
    feedbacks = data.get('feedbacks', [])

    if not feedbacks:
        return jsonify({'error': 'Please provide a list of feedback texts.'}), 400

    results = []
    for text in feedbacks:
        text = text.strip()
        if text:
            result = analyze_sentiment(text)
            if result:
                results.append({
                    'feedback': text,
                    'sentiment': result['sentiment'],
                    'confidence': result['confidence']
                })

    add_history(results)

    positive_count = sum(1 for r in results if r['sentiment'] == 'positive')
    negative_count = sum(1 for r in results if r['sentiment'] == 'negative')
    neutral_count = sum(1 for r in results if r['sentiment'] == 'neutral')
    total = len(results) if results else 1

    summary = {
        'positive': positive_count,
        'negative': negative_count,
        'neutral': neutral_count,
        'positive_pct': round(positive_count / total * 100, 1),
        'negative_pct': round(negative_count / total * 100, 1),
        'neutral_pct': round(neutral_count / total * 100, 1),
        'total': len(results)
    }

    return jsonify({'results': results, 'summary': summary})


@app.route('/history')
@login_required
def history():
    history = get_history()
    stats = get_dashboard_stats()
    return render_template('history.html', page='history', history=history,
                           stats=stats, user=session['user'])


@app.route('/api/history_stats')
@login_required
def history_stats():
    return jsonify(get_dashboard_stats())


if __name__ == '__main__':
    load_model()
    app.run(debug=False, host='0.0.0.0', port=5500)
