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
When I study in my school, I recognized that my school system has the function for the student to submit their researches or documents, but they cannot define whether their documents can be copied or not.
Therefore, My purpose is to build a library system with the function of storing documents such as research articles, reports, slides, to serve the preservation of user documents. The topic can apply AI in classifying the content of the document, thereby checking and considering whether the document is copied from previous documents and at the same time displaying the similarity between the documents and notify users to be aware of content copying.
To define target user and market. There are three main markets and users in the system.
'''

# Build the model
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
# Train the model using the training data
model.fit(train_data.data, train_data.target)
# Predict the categories of the test data
predicted_categories = model.predict([my_sentence])

all_categories_names = np.array(data.target_names)

print(all_categories_names[predicted_categories])
