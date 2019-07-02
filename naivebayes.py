import pandas as pd

df = pd.read_table('/home/sagar/Downloads/SMSSpamCollection',sep='\t',header=None,names=['label', 'message'])

#pre-processing
df['label'] = df.label.map({'ham': 0, 'spam': 1})
df['message'] = df.message.map(lambda x: x.lower())
df['message'] = df.message.str.replace('[^\w\s]', '')
import nltk
#nltk.download()
df['message'] = df['message'].apply(nltk.word_tokenize)
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
df['message'] = df['message'].apply(lambda x: [stemmer.stem(y) for y in x])

from sklearn.feature_extraction.text import CountVectorizer
# This converts the list of words into space-separated strings
df['message'] = df['message'].apply(lambda x: ' '.join(x))
count_vect = CountVectorizer()
counts = count_vect.fit_transform(df['message'])
print(type(counts))
print(counts)
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer().fit(counts)
counts = transformer.transform(counts)

#training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(counts, df['label'], test_size=0.1, random_state=69)

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB().fit(X_train, y_train)

#evaluating
import numpy as np
predicted = model.predict(X_test)
print(np.mean(predicted == y_test))

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test, predicted))
print(classification_report(y_test, predicted))
