import pickle
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

# import pandas, xgboost, numpy, textblob, string
from tensorflow.python.keras.preprocessing import text, sequence
from tensorflow.python.keras import layers, models, optimizers
from tensorflow.python.keras.layers import *


from pyvi import ViTokenizer, ViPosTagger
from tqdm import tqdm
import numpy as np
import gensim
import numpy as np

import os
dir_path = os.path.dirname(os.path.realpath(os.getcwd()))
dir_path = os.path.join(dir_path, 'Data')
# '/Users/macos/Desktop/Github/NLP/Text Classifier'
# Load data from dataset folder
# VNTC-master/Data/10Topics/Ver1.1/Train_Full
# VNTC-master/Data/10Topics/Ver1.1/Test_Full


def get_data(folder_path):
    X = []
    y = []
    dirs = os.listdir(folder_path)
    for path in dirs:
        file_paths = os.listdir(os.path.join(folder_path, path))
        for file_path in tqdm(file_paths):
            with open(os.path.join(folder_path, path, file_path), 'r', encoding="utf-16") as f:
                lines = f.readlines()
                lines = ' '.join(lines)
                lines = gensim.utils.simple_preprocess(lines)
                lines = ' '.join(lines)
                lines = ViTokenizer.tokenize(lines)
#                 sentence = ' '.join(words)
#                 print(lines)
                X.append(lines)
                y.append(path)
#             break
#         break
    return X, y


train_path = os.path.join(
    dir_path, 'VNTC-master\\Data\\27Topics\\Ver1.1\\Train_Full')
X_data, y_data = get_data(train_path)


pickle.dump(X_data, open('data/X_data.pkl', 'wb'))
pickle.dump(y_data, open('data/y_data.pkl', 'wb'))

test_path  = os.path.join(
    dir_path, 'VNTC-master\\Data\\27Topics\\Ver1.1\\Test\\Test')
X_test, y_test = get_data(test_path)


pickle.dump(X_test, open('data/X_test.pkl', 'wb'))
pickle.dump(y_test, open('data/y_test.pkl', 'wb'))