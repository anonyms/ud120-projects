#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list,sort_keys = '../tools/python2_lesson14_keys.pkl')
labels, features = targetFeatureSplit(data)



### your code goes here 
from sklearn import tree
from sklearn.cross_validation import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split( features, labels, test_size=0.3, random_state=42)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train,labels_train)
print "the inital accuracy is : ", clf.score(features_test,labels_test)

##Inital questions
predictions = clf.predict(features_test)
print "Number of POIs from test set : ",predictions.sum()
print "People in test set : ",predictions.size
import numpy as np 
labels_test_np = np.array(labels_test)
predictions_all_zero = predictions 
predictions_all_zero[predictions_all_zero == 1.0 ] = 0.0
print "New accuracy if predictions all zero : ",  (np.sum(predictions_all_zero == labels_test_np))/( float(predictions_all_zero.size) )
from sklearn import metrics as metrics
print "The precision is : ", metrics.precision_score(labels_test,predictions)
print "The recall is : ", metrics.recall_score(labels_test,predictions)
