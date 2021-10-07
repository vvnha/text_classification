import pickle
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from pyvi import ViTokenizer, ViPosTagger
from tqdm import tqdm
import numpy as np
import gensim
import numpy as np
from tensorflow.python.keras import models
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
from tensorflow.python.keras.optimizer_v2.adam import Adam

file_model = "dataE/model.json"

# loaded_model = models.load_model("data/model")

loaded_model = joblib.load("dataE/model.joblib")


def preprocessing_doc(doc):
    lines = gensim.utils.simple_preprocess(doc)
    lines = ' '.join(lines)
    lines = ViTokenizer.tokenize(lines)

    return lines


test_doc = '''build a library system for saving materials
'''

X_data = pickle.load(open('dataE/X_data.pkl', 'rb'))
y_data = pickle.load(open('dataE/y_data.pkl', 'rb'))

tfidf_vect = pickle.load(open('dataE/tfidf_vect.pkl', 'rb'))
svd = joblib.load("dataE/svd.p")

test_doc = preprocessing_doc(test_doc)
test_doc_tfidf = tfidf_vect.transform([test_doc])
print(np.shape(test_doc_tfidf))
test_doc_svd = svd.transform(test_doc_tfidf)
probality = loaded_model.predict_proba(test_doc_tfidf)

encoder = preprocessing.LabelEncoder()
y_data_n = encoder.fit_transform(y_data)
typeText = encoder.classes_

rs = loaded_model.predict(test_doc_tfidf)
rsArray = list(zip(probality.flatten(), typeText))


def takeFirst(elem):
    return elem[0]


rsArray.sort(key=takeFirst, reverse=True)

print(rsArray)
