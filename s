Project uses Python. Here's the app component
from flask import Flask, render_template
from fetch_articles import fetch_articles
from sentiment_analysis import analyze_sentiment, sentiment_labels

app = Flask(name)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results')
def results():
    articles = fetch_articles()
    results = analyze_sentiment(articles)
    return render_template('results_page.html', results=results, sentiment_labels=sentiment_labels)

if name == 'main':
    app.run(debug=True)

here's the fetch article component:
import os
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup

# Load environment variables from .env file
load_dotenv()

# Retrieve API key from environment variables
api_key = os.getenv('GUARDIAN_API_KEY')

# Base URL for The Guardian API
base_url = 'https://content.guardianapis.com/search'

# Timezone conventions to exclude
exclude_timezones = ['UTC', 'GMT', 'BST', 'PDT', 'EDT', 'CET', 'AEDT', 'IST', 'JST', 'CST', 'KST', 'MSK']

# Parameters for the API request
params = {
    'api-key': api_key,
    'show-fields': 'body',  # Request to show full article body
    'order-by': 'newest',  # Order by newest articles
    'production-office': 'aus',  # Filter by Australian edition
    'page-size': 20  # Number of results per page
}

# Function to fetch articles
def fetch_articles():
    articles_list = []

    try:
        # Making the request
        response = requests.get(base_url, params=params)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            data = response.json()

            # Process the data here
            for article in data.get('response', {}).get('results', []):
                headline = article.get('webTitle', '')
                body = article.get('fields', {}).get('body', '')

                # Exclude articles with 'GMT' in the body text or timezones in exclude_timezones
                exclude_article = any(tz in body for tz in exclude_timezones) or 'GMT' in body

                if not exclude_article:
                    # Parse HTML content to extract plain text
                    soup = BeautifulSoup(body, 'html.parser')
                    full_text = soup.get_text()

                    # Create a dictionary for each article
                    article_dict = {
                        'headline': headline,
                        'full_text': full_text
                    }

                    # Append dictionary to articles list
                    articles_list.append(article_dict)

        else:
            print(f"Request failed with status code {response.status_code}")
            print(response.text)

    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")

    return articles_list

Here's the sentiment analysis component
import torch
from transformers import BertForSequenceClassification, BertTokenizer
import spacy
from collections import Counter
from fetch_articles import fetch_articles

# Load tokenizer and model for sentiment analysis
tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')

# Load spaCy for NER
nlp = spacy.load('en_core_web_sm')

# Sentiment labels
sentiment_labels = {
    0: 'positive',
    1: 'negative',
    2: 'neutral'
}

# Function to chunk input_ids and attention_mask
def get_input_ids_and_attention_mask_chunk(tokens, chunksize=510):
    input_ids = tokens['input_ids'][0]
    attention_mask = tokens['attention_mask'][0]

    input_id_chunks = []
    attention_mask_chunks = []

    for i in range(0, len(input_ids), chunksize):
        input_chunk = input_ids[i:i+chunksize]
        attention_chunk = attention_mask[i:i+chunksize]

        # Add [CLS] and [SEP] tokens at the beginning and end of each chunk
        input_chunk = torch.cat([torch.tensor([101]), input_chunk, torch.tensor([102])])
        attention_chunk = torch.cat([torch.tensor([1]), attention_chunk, torch.tensor([1])])

        # Pad if necessary
        padding_length = chunksize + 2 - input_chunk.shape[0]
        if padding_length > 0:
            input_chunk = torch.cat([input_chunk, torch.zeros(padding_length)])
            attention_chunk = torch.cat([attention_chunk, torch.zeros(padding_length)])

        input_id_chunks.append(input_chunk.unsqueeze(0))
        attention_mask_chunks.append(attention_chunk.unsqueeze(0))

    return input_id_chunks, attention_mask_chunks

# Function to perform sentiment analysis
def analyze_sentiment(articles_list):
    results = []

    for article in articles_list:
        # Extract text and headline
        text = article['full_text']
        headline = article['headline']

        # Tokenize the text
        tokens = tokenizer.encode_plus(text, add_special_tokens=False, return_tensors='pt')

        # Get input_ids and attention_mask chunks
        input_id_chunks, attention_mask_chunks = get_input_ids_and_attention_mask_chunk(tokens)

        # Perform inference for each chunk and accumulate results
        total_probabilities = None
        chunk_sentiments = []
        chunk_probabilities = []
        chunk_texts = []

        for idx, (input_ids_chunk, attention_mask_chunk) in enumerate(zip(input_id_chunks, attention_mask_chunks)):
            # Prepare input dictionary
            input_dict = {
                'input_ids': input_ids_chunk.long(),
                'attention_mask': attention_mask_chunk.int()
            }

            # Perform inference
            outputs = model(**input_dict)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=-1)

            # Predict sentiment for the chunk
            predicted_sentiment = torch.argmax(probabilities).item()
            chunk_sentiments.append(predicted_sentiment)
            chunk_probabilities.append(probabilities.tolist())

            # Extract chunk text
            chunk_text = tokenizer.decode(input_ids_chunk[0], skip_special_tokens=True)
            chunk_texts.append(chunk_text)

            # Accumulate probabilities for mean calculation
            if total_probabilities is None:
                total_probabilities = probabilities
            else:
                total_probabilities += probabilities

        # Calculate mean probabilities
        mean_probabilities = total_probabilities / len(input_id_chunks)

        # Predicted sentiment for the entire text is the argmax of mean probabilities
        predicted_sentiment = torch.argmax(mean_probabilities).item()

        # Overall predicted sentiment
        overall_sentiment_label = sentiment_labels[predicted_sentiment]

        # Extract entities from the text
        doc = nlp(text)
        entities = [ent.text for ent in doc.ents if ent.label_ in ['ORG', 'PERSON', 'GPE']]

        # Count occurrences of each entity
        entity_counts = Counter(entities)
        most_common_entities = entity_counts.most_common(2)

        # Create results dictionary
        result_dict = {
            'headline': headline,
            'overall_sentiment': overall_sentiment_label,
            'chunk_probabilities': chunk_probabilities,
            'chunk_texts': chunk_texts,
            'chunk_sentiments': chunk_sentiments,
            'mean_probabilities': mean_probabilities.tolist(),  # Convert tensor to list for JSON serialization
            'most_common_entities': most_common_entities
        }

        # Append results dictionary to results list
        results.append(result_dict)

    return results

index.html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentiment Analyzer</title>
    <link href="{{ url_for('static', filename='css/tailwind.css') }}" rel="stylesheet">
</head>

<body class="bg-gray-100 flex items-center justify-center h-screen">
    <div class="bg-white rounded-lg p-8 shadow-lg">
        <h1 class="text-3xl font-bold mb-4">Welcome to Sentiment Analyzer</h1>
        <a href="{{ url_for('results') }}" class="bg-blue-500 hover:bg-blue-600 text-white py-2 px-4 rounded-lg inline-block">
            Calculate Sentiment
        </a>
    </div>
</body>

</html>

and results page
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Results Page</title>
</head>
<body>
    <h1>Results Page</h1>

    {% for result in results %}
    <div>
        <h2>{{ result.headline }}</h2>
        <p><strong>Overall Sentiment:</strong> {{ result.overall_sentiment }}</p>

        <h3>Probability Distributions</h3>

        <ul>
            {% for chunk_prob in result.chunk_probabilities %}
            <li>
                <p><strong>Sentiment:</strong> {{ sentiment_labels[result.chunk_sentiments[loop.index0]] }}</p>
                <p><strong>Probabilities:</strong></p>
                <ul>
                    {% for prob in chunk_prob %}
                    <li>{{ prob }}</li>
                    {% endfor %}
                </ul>
            </li>
            {% endfor %}
        </ul>

        <p><strong>Overall Probability:</strong></p>
        <ul>
            {% for prob in result.mean_probabilities %}
            <li>{{ prob }}</li>
            {% endfor %}
        </ul>

        <p><strong>Overall Sentiment:</strong> {{ result.overall_sentiment }}</p> <!-- Moved this line here -->

        <hr>
    </div>
    {% endfor %}
</body>
</html>