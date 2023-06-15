import pandas as pd
import json
import random
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load the JSON dataset
dataset_path = "idmanual.json"
with open(dataset_path, 'r') as file:
    json_data = json.load(file)

# Extract description and class from the dataset
descriptions = [data["description"] for data in json_data]
classes = [data["class_id"] for data in json_data]

# Preprocess the text data
lemmatizer = WordNetLemmatizer()
stopwords_set = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
    tokens = [token for token in tokens if token not in stopwords_set]
    return tokens

preprocessed_descriptions = [preprocess_text(desc) for desc in descriptions]

# Split the dataset into training and testing sets
train_descriptions, test_descriptions, train_classes, test_classes = train_test_split(
    preprocessed_descriptions,
    classes, test_size=0.2, random_state=42)

# Print sample data
print("Sample preprocessed description:", train_descriptions[0])
print("Sample class:", train_classes[0])