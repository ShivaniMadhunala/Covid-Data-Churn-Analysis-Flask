# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 10:19:32 2020

@author: administrator
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
# Importing the dataset
dataset = pd.read_csv('data_large.csv',header=None,delimiter=',')
X = dataset.iloc[:,0:15]     
y = dataset.iloc[:, 15].values
corr = X.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)
sns.heatmap(corr,annot=True,fmt=".2f")
plt.show()

#datapreprocessing


#sum_of_diseases
X1 = X.iloc[:,[8,9,10,11]].values
Y1=list(X1.sum(axis=1))
X=pd.DataFrame(X.drop([8,9,10,11], axis=1).values)

X2=np.array([Y1])           
X=X.transpose()  
X=np.append(X,X2,axis=0)        
X=X.transpose()
  
# Encoding categorical data
labelencoder_X_1 = LabelEncoder()
X[:, 0] = labelencoder_X_1.fit_transform(X[:, 0])

transformer = ColumnTransformer(
    transformers=[
        ("OneHot",        # Just a name
         OneHotEncoder(), # The transformer class
         [0]              # The column(s) to be applied on.
         )
    ],
    remainder='passthrough' # donot apply anything to the remaining columns
)
X = transformer.fit_transform(X.tolist())
X = X.astype('float64')
X = X[:, 1:]
X=np.array(X)



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

test_file = X_train[0:15,:]

np.savetxt('test_file.csv',test_file, fmt="%.6g" , delimiter=',')

import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu', input_dim = 13))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 32, nb_epoch = 50)



# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



import pickle
pickle.dump(classifier, open("model.pkl","wb"))

import seaborn as sn
sn.heatmap(cm,annot=True,fmt="d")

