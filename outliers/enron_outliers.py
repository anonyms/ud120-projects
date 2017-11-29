#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]
data_dict.pop('TOTAL',0)
data = featureFormat(data_dict, features)


list = data_dict.keys()

for x in list: 
	if not(data_dict[x]['salary'] == 'NaN') and data_dict[x]['salary'] > 1000000:
		if not(data_dict[x]['bonus'] == 'NaN') and data_dict[x]['bonus'] > 5000000:
			print "Key ",x," salary: ",data_dict[x]['salary'],"  bonus: ",data_dict[x]['bonus']

### your code below

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

