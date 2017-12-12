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
feature_3 = "total_payments"
poi  = "poi"
features_list = [poi, feature_1, feature_2, feature_3]
data = featureFormat(data_dict, features_list )
poi, finance_features = targetFeatureSplit( data )

list = data_dict.keys()
max_eso = 0
min_eso = 10000000
maxer = ''
miner = ''

for x in list: 

    if(not(data_dict[x]['exercised_stock_options'] == 'NaN')):

        if(data_dict[x]['exercised_stock_options'] > max_eso):
            max_eso = data_dict[x]['exercised_stock_options']
            maxer = x
        elif(data_dict[x]['exercised_stock_options'] < min_eso):
            min_eso = data_dict[x]['exercised_stock_options']
            miner = x


print "Maximum: ",max_eso," Who you say ? This SOB : ",maxer
print "Minimum: ",min_eso," Who you say ? This SOB : ",miner


list = data_dict.keys()
max_sal = 0
min_sal = 10000000
maxer = ''
miner = ''

for x in list: 

    if(not(data_dict[x]['salary'] == 'NaN')):

        if(data_dict[x]['salary'] > max_sal):
            max_sal = data_dict[x]['salary']
            maxer = x
        elif(data_dict[x]['salary'] < min_sal):
            min_sal = data_dict[x]['salary']
            miner = x


print "Maximum: ",max_sal," Who you say ? This SOB : ",maxer
print "Minimum: ",min_sal," Who you say ? This SOB : ",miner