#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
##i = 0
##for p in enron_data:
##	if(enron_data[p]['poi']==1): i = i + 1
list = enron_data.keys()
count_people = 0
count_emails = 0
count_salary = 0
count_payments = 0
count_poi_nanpayments = 0
count_poi = 0
for x in list: 
	count_people = count_people + 1
	if(not(enron_data[x]['salary'] == 'NaN')):
		count_salary = count_salary+ 1
	if(not(enron_data[x]['email_address'] == 'NaN')):
		count_emails = count_emails +1
	if(not(enron_data[x]['total_payments'] == 'NaN')):
		count_payments = count_payments +1
	if(enron_data[x]['poi']):
		count_poi = count_poi + 1
		if(enron_data[x]['total_payments'] == 'NaN'):
			count_poi_nanpayments = count_poi_nanpayments + 1

print "People : ",count_people," Emails : ",count_emails," Salaries : ", count_salary, " Total Payements : ", count_payments
print "POIs : ", count_poi," Pois without payments : ", count_poi_nanpayments

## In order to test and use featureFormat functions
## = ['salary','total_payments']
##test = featureFormat(enron_data,features)
##print test