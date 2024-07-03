import torch
from transformers import BertForSequenceClassification, BertTokenizer
import spacy
from collections import Counter

# Import the fetch_articles function from fetch_articles.py
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

# List to store results
results = []

# Fetch articles
articles_list = fetch_articles()

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

        # Extract chunk text (for demonstration purposes, not exact text)
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
        'most_common_entities': most_common_entities,
        # 'num_chunks': len(input_id_chunks),  # Add number of chunks
        # 'chunk_probabilities': chunk_probabilities,
        # 'chunk_texts': chunk_texts  # Add chunk texts
    }

    # Append results dictionary to results list
    results.append(result_dict)

# Print the results
for result in results:
    print(f"Headline: {result['headline']}")
    print(f"Overall Sentiment: {result['overall_sentiment']}")
    print(f"Most Common Entities: {result['most_common_entities']}")
    # print(f"Number of Chunks: {result['num_chunks']}")
    # for idx, (chunk_text, chunk_prob) in enumerate(zip(result['chunk_texts'], result['chunk_probabilities'])):
    #     print(f"Chunk {idx + 1} Text: {chunk_text[:100]}...")  # Print first 100 characters of the chunk text
    #     print(f"Chunk {idx + 1} Probabilities: {chunk_prob}")
    print("---------------------------")