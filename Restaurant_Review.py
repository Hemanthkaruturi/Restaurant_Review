#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 19:25:07 2018

@author: darling
"""

#Restaurant Review
#NLTK Project
import pandas as pd
import numpy as np

#Importing data
data = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

#Cleaning the data
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []                                                                         #Any kind of text generally called corpus
for i in range(0,len(data)):
    review = re.sub('[^a-zA-Z]',' ',data['Review'][i])                                  #Removing all the characters except a to z
    review = review.lower()                                                             #Converting Capital Letters into lower case
    review = review.split()                                                             #Splitting sentence into individual words
    ps = PorterStemmer()                                                                #Stemming   (love = loved, loving, lovable, lover, lovely, ...)
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]   #Removing the words present in stopwords
    review = " ".join(review)                                                            #Joining the individual words to a sentence)
    corpus.append(review)             
    
#Creating the Bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()          #This will create a sparse matix of zero's(count on number of times word repeated) and then it converts into array    
y = data.iloc[:, 1].values                      #Taking Target variable

#Building the machine learning model

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
rf_classifier.fit(X_train, y_train)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dt_classifier.fit(X_train, y_train)


####################### ANN ########################################
# Importing the Keras libraries and packages
# Installing Tensorflow
#pip install tensorflow

# Installing Keras
# pip install --upgrade keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
ann_classifier = Sequential()

# Adding the input layer and the first hidden layer
ann_classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 1500))

# Adding the second hidden layer
ann_classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
ann_classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
ann_classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
ann_classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)


##########################################################################


# Predicting the Test set results
rf_pred = rf_classifier.predict(X_test)
nb_pred = nb_classifier.predict(X_test)
dt_pred = dt_classifier.predict(X_test)

ann_pred = ann_classifier.predict(X_test)
ann_pred = (ann_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_rf = confusion_matrix(y_test, rf_pred)                       #72% accuracy
cm_nb = confusion_matrix(y_test, nb_pred)                       #73% accuracy
cm_dt = confusion_matrix(y_test, dt_pred)                       #71% accuracy
cm_ann = confusion_matrix(y_test, ann_pred)                     #71.5% accuracy