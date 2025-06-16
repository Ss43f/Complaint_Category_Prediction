

from flask import Flask, request, render_template_string
import joblib, re, string, nltk
from nltk.corpus import stopwords

# Ensure stopwords are downloaded
try:
    stopwords = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stopwords = set(stopwords.words('english'))

try:
    model = joblib.load('Logistic_Regression_best_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
except FileNotFoundError:
    print("❌ Model or vectorizer file not found.")
    model, vectorizer = None, None

app = Flask(__name__)

html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Complaint Classifier</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f0f2f5; padding: 50px; }
        .container { background: white; padding: 20px; border-radius: 8px; max-width: 600px; margin: auto; box-shadow: 0px 0px 10px rgba(0,0,0,0.1); }
        textarea { width: 100%; height: 150px; margin-top: 10px; padding: 10px; font-size: 16px; }
        button { margin-top: 10px; padding: 10px 20px; font-size: 16px; background-color: #007BFF; color: white; border: none; border-radius: 5px; }
        h2 { color: #333; }
        .result { margin-top: 20px; font-size: 18px; font-weight: bold; color: #007BFF; }
    </style>
</head>
<body>
    <div class="container">
        <h2>Complaint Category Prediction</h2>
        <form method="post">
            <textarea name="text" placeholder="Enter your complaint here..."></textarea><br>
            <button type="submit">Predict</button>
        </form>
        {% if prediction %}
            <div class="result">Predicted Category: {{ prediction }}</div>
        {% endif %}
    </div>
</body>
</html>
"""

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords]
    return ' '.join(tokens)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST' and model and vectorizer:
        text = request.form['text']
        cleaned = clean_text(text)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
    return render_template_string(html_template, prediction=prediction)

if __name__ == '__main__':
    print("🌐 Web app running at http://127.0.0.1:5000/")
    app.run(debug=True)

