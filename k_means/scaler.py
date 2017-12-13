#!/usr/bin/python 

""" 
    Skeleton code for k-means clustering mini-project.
"""




import pickle
import numpy
import matplotlib.pyplot as plt
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)


### the input features we want to use 
### can be any key in the person-level dictionary (salary, director_fees, etc.) 
feature_1 = "salary"
feature_2 = "exercised_stock_options"
##feature_3 = "total_payments"
poi  = "poi"
features_list = [poi, feature_1, feature_2]
data = featureFormat(data_dict, features_list )
poi, finance_features = targetFeatureSplit( data )

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
eso_scaler = MinMaxScaler()


scaled = scaler.fit_transform(finance_features)
min_sal = 100000
max_sal = 0
min_eso = 100000
max_eso = 0

for f1,f2 in finance_features:
    if f1<min_sal:
        min_sal = f1
    elif f1 > max_sal:
        max_sal = f1

    if f2<min_eso:
        min_eso = f2
    elif f2 > max_eso:
        max_eso = f2

print "min sal is: ", min_sal, " max sal is: ",max_sal
print "min eso is: ", min_eso, " max eso is: ",max_eso

scalval200 = (200000. - min_sal)/(max_sal - min_sal)
scalval1000000 = (1000000. - min_eso)/(max_eso - min_eso)

print "So 200k sal is: ", scalval200, "And the other motherfucker is: ",scalval1000000




