from datetime import datetime

import os
from tensorflow.python.keras.preprocessing import text, sequence
from tensorflow.python.keras import layers, models, optimizers
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.optimizer_v2.adam import Adam

from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn import model_selection, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.externals import joblib
import pickle

import gensim
import numpy as np
from tqdm import tqdm
from pyvi import ViTokenizer, ViPosTagger


X_data = pickle.load(open('data/X_data.pkl', 'rb'))
y_data = pickle.load(open('data/y_data.pkl', 'rb'))

X_test = pickle.load(open('data/X_test.pkl', 'rb'))
y_test = pickle.load(open('data/y_test.pkl', 'rb'))

tfidf_vect = TfidfVectorizer(analyzer='word', max_features=30000)
tfidf_vect.fit(X_data)  # learn vocabulary and idf from training set
X_data_tfidf = tfidf_vect.transform(X_data)
X_test_tfidf =  tfidf_vect.transform(X_test)

svd = TruncatedSVD(n_components=300, random_state=42)
svd.fit(X_data_tfidf)

X_data_tfidf_svd = svd.transform(X_data_tfidf)
X_test_tfidf_svd = svd.transform(X_test_tfidf)

filetdif_vect = "data/tfidf_vect.pkl"
pickle.dump(tfidf_vect, open(filetdif_vect, "wb"))

fileSvd = "data/svd.p"
joblib.dump(svd, fileSvd)

# X_data_tfidf_svd = svd.transform(X_data_tfidf)

encoder = preprocessing.LabelEncoder()
y_data_n = encoder.fit_transform(y_data)
y_test_n = encoder.fit_transform(y_test)

def preprocessing_doc(doc):
    lines = gensim.utils.simple_preprocess(doc)
    lines = ' '.join(lines)
    lines = ViTokenizer.tokenize(lines)

    return lines

def train_model(classifier, X_data, y_data, X_test=None, y_test=None, is_neuralnet=False, n_epochs=3):
    X_train, X_val, y_train, y_val = train_test_split(
        X_data, y_data, test_size=0.1, random_state=42)
    filename = "data/model"
    fileXdata = "data/X_data_tfidf.pkl"

    if is_neuralnet:
        classifier.fit(X_train, y_train, validation_data=(
            X_val, y_val), epochs=n_epochs, batch_size=512)

        val_predictions = classifier.predict(X_val)
        test_predictions = classifier.predict(X_test)
        val_predictions = val_predictions.argmax(axis=-1)
#         test_predictions = test_predictions.argmax(axis=-1)
    else:
        classifier.fit(X_train, y_train)

        train_predictions = classifier.predict(X_train)
        val_predictions = classifier.predict(X_val)
#         test_predictions = classifier.predict(X_test)


    classifier.save(filename)
    pickle.dump(X_data, open(fileXdata, 'wb'))
    print("Validation accuracy: ", metrics.accuracy_score(val_predictions, y_val))


def create_rcnn_model():
    input_layer = Input(shape=(300,))
    
    layer = Reshape((10, 30))(input_layer)
    layer = Bidirectional(GRU(128, activation='relu', return_sequences=True))(layer)    
    layer = Convolution1D(100, 3, activation="relu")(layer)
    layer = Flatten()(layer)
    layer = Dense(512, activation='relu')(layer)
    layer = Dense(512, activation='relu')(layer)
    layer = Dense(128, activation='relu')(layer)
    
    output_layer = Dense(27, activation='softmax')(layer)
    
    classifier = models.Model(input_layer, output_layer)
    classifier.summary()
    classifier.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return classifier

model = create_rcnn_model()
# train_model(model, X_data_tfidf, y_data_n, , is_neuralnet=False)
train_model(classifier=model, X_data=X_data_tfidf_svd, y_data=y_data_n, X_test=X_test_tfidf_svd, y_test=y_test_n, is_neuralnet=True, n_epochs=20)

test_doc = '''	
Tài trợ thương mại Quốc tế và một số giải pháp để nâng cao hiệu quả hoạt động tài trợ thương mại Quốc tế của ngân hàng công thương Việt Nam'''


test_doc_tfidf = tfidf_vect.transform([test_doc])
print(np.shape(test_doc_tfidf))
test_doc_svd = svd.transform(test_doc_tfidf)

print(model.predict(test_doc_tfidf))

# print("TensorFlow version: ", tf.__version__)
