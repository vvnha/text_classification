import numpy as np
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
# sns.set()  # use seaborn plotting style

# Load the dataset
data = fetch_20newsgroups()
# Get the text categories
text_categories = data.target_names
# define the training set
train_data = fetch_20newsgroups(subset="train", categories=text_categories)
# define the test set
test_data = fetch_20newsgroups(subset="test", categories=text_categories)

my_sentence = ''''
I will build a system for users to save their materials with the main features is automatically classifying documents and check the similarity of the document to avoid copying the text
'''

# Build the model
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
# Train the model using the training data
model.fit(train_data.data, train_data.target)
# Predict the categories of the test data
predicted_categories = model.predict([my_sentence])

all_categories_names = np.array(data.target_names)

print(all_categories_names[predicted_categories])
