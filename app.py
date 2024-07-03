from flask import Flask, render_template
from fetch_articles import fetch_articles
from sentiment_analysis import analyze_sentiment

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results')
def results():
    articles = fetch_articles()
    results = analyze_sentiment(articles)
    return render_template('results_page.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)