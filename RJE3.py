import pandas as pd
import os
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from bs4 import BeautifulSoup
from sklearn.neural_network import MLPClassifier
import pickle
from nltk.tokenize import word_tokenize
import sys



#punct
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
	
	text = BeautifulSoup(text, "lxml").text # HTML decoding
	text = text.lower() # lowercase text
	text = REPLACE_BY_SPACE_RE.sub('', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
	text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
	text = ' '.join(word for word in text.split() if word not in STOPWORDS)# delete stopwors from text

	stemmer = PorterStemmer()
	words = stopwords.words("english")
	word_tokens = word_tokenize(text)  
	filtered_sentence = [w for w in word_tokens if not w in words]  
	filtered_sentence = []  
	for w in word_tokens:  
		if w not in words:  
			filtered_sentence.append(w)  

	text_clean = ""
	for w in filtered_sentence:
		text_clean += stemmer.stem(w) + ' '

	return [text_clean]

def find_TA(text):

	THIS_FOLDER = str(os.path.dirname(os.path.abspath(__file__)))
	loaded_clf = pickle.load(open(THIS_FOLDER+'/E3classifier.pickle', 'rb'))
	X = clean_text(text)
	predictions = loaded_clf.predict(X)
	return(predictions)

def main():
	my_text = str(sys.argv[1])
	predictions = find_TA(my_text)
	sys.stdout.write(str(predictions[0]))
	

if __name__ == "__main__":
	#app.run()
	main()

