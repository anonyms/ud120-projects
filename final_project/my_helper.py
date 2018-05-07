#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

##When testing
## from my_helper import computeFraction,draw2FeaturesinScatter,drawmultiplescatters,getDataDict,getFractionToAndFromPoi,getWho

def computeFraction( poi_messages, all_messages ):
	""" given a number messages to/from POI (numerator) 
		and number of all messages to/from a person (denominator),
		return the fraction of messages to/from that person
		that are from/to a POI
   """


	### you fill in this code, so that it returns either
	###     the fraction of all messages to this person that come from POIs
	###     or
	###     the fraction of all messages from this person that are sent to POIs
	### the same code can be used to compute either quantity

	### beware of "NaN" when there is no known email address (and so
	### no filled email features), and integer division!
	### in case of poi_messages or all_messages having "NaN" value, return 0.
	fraction = 0.
	if all_messages != 0 and all_messages != "NaN" and poi_messages != "NaN":
		fraction = float(poi_messages)/float(all_messages)


	return fraction

def draw2FeaturesinScatter(features, pois, indexfeature1 = 0, indexfeature2 = 1, name_feature1 = "feature 1", name_feature2 = "feature 2", prefix = ""):
	i = 0
	for point in features:
		features1 = point[indexfeature1]
		features2 = point[indexfeature2]
		if pois[i] == 1:
			plt.scatter( features1, features2, color = "r")
		else:
			plt.scatter( features1, features2, color = "b")
		i = i+1

	plt.xlabel(name_feature1)
	plt.ylabel(name_feature2)
	name = prefix + name_feature1 + "_AND_" +name_feature2
	plt.savefig("scatters/"+name)
	plt.close()
	##plt.show()

def drawMatrixPlot(features_list):
	import pandas as pd
	from matplotlib.colors import ListedColormap
	with open("final_project_dataset.pkl", "r") as data_file:
		data_dict = pickle.load(data_file)

	### Task 2: Remove outliers
	del data_dict['TOTAL'] ##Only outlier remover for now
	### Task 3: Create new feature(s)
	data_dict = getFractionToAndFromPoi(data_dict)
	data = featureFormat(data_dict, features_list)
	labels, features = targetFeatureSplit(data)

	# Note: It appears that pandas.scatter_matrix doesn't quite work
	#       as advertised, in the documentation. If it did, this wouldn't
	#       be necessary. You could pass a colormap, instead.
	palette = {0 : 'blue', 1 : 'red'}
	labels_c = map(lambda x: palette[int(x)], labels)

	data_frame = pd.DataFrame(features, columns=features_list[1:])
	grr = pd.scatter_matrix(data_frame, alpha=0.8, c=labels_c)
	plt.savefig("scatters/MatrixPlot")
	plt.close()

def drawmultiplescatters(features_list = [], data_dict = []):

	## The features to be drawn
	if features_list == []:
		features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 
		'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',
		'to_messages','from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] 
		## Load the dictionary
		with open("final_project_dataset.pkl","r") as data_file:
			data_dict = pickle.load(data_file)
		del data_dict['TOTAL']
	data = featureFormat(data_dict, features_list, sort_keys = True)
	labels, features = targetFeatureSplit(data)
	## Run over all features, and create scatter
	for i_f1 in range(0,len(features_list)-1):
		for j_f2 in range(0,len(features_list)-1):
			if i_f1 > j_f2:
				draw2FeaturesinScatter(features,labels,indexfeature1 = i_f1, indexfeature2 = j_f2, name_feature1 = features_list[i_f1+1], name_feature2 = features_list[j_f2+1])

def getDataDict():
	with open("final_project_dataset.pkl", "r") as data_file:
		data_dict = pickle.load(data_file)
	return data_dict

def getFractionToAndFromPoi(data_dict):
	submit_dict = {}
	for name in data_dict:

		data_point = data_dict[name]

		from_poi_to_this_person = data_point["from_poi_to_this_person"]
		to_messages = data_point["to_messages"]
		fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
		data_point["fraction_from_poi"] = fraction_from_poi


		from_this_person_to_poi = data_point["from_this_person_to_poi"]
		from_messages = data_point["from_messages"]
		fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
		submit_dict[name]={"from_poi_to_this_person":fraction_from_poi,
						   "from_this_person_to_poi":fraction_to_poi}
		data_point["fraction_to_poi"] = fraction_to_poi

	return data_dict


def getMax(data_dict, feature):
	maxvalue = 0
	for name in data_dict:
		person_data = data_dict[name]
		if person_data[feature] != 'NaN':
			if person_data[feature] > maxvalue:
				maxvalue = person_data[feature]

	return maxvalue

def getScaledFeature(data_dict,feature):
	submit_dict = {}
	maxval = getMax(data_dict,feature)
	new_feature_name = "scaled_"+feature
	for name in data_dict:
		person_data = data_dict[name]
		this_person_val = person_data[feature]
		scaled_feature = computeFraction( this_person_val, maxval )
		person_data[new_feature_name] = scaled_feature

	return data_dict

def getWho(data_dict,feature,value,operator = "="):
	res = []
	for name in data_dict:

		data_point = data_dict[name]
		if operator == "=":
			if data_point[feature] != 'NaN' and data_point[feature] == value:
				res.append(name)
		elif data_point[feature] != 'NaN' and operator == ">=":
			if data_point[feature] >= value:
				res.append(name)
		elif data_point[feature] != 'NaN' and operator == "<=":
			if data_point[feature] <= value:
				res.append(name)

	return res

def useSkbestToRateFeatures():
	features_list = ['poi', 'salary','to_messages','deferral_payments','total_payments','exercised_stock_options','bonus','restricted_stock','shared_receipt_with_poi','restricted_stock_deferred','total_stock_value','expenses','loan_advances','from_messages','other','from_this_person_to_poi','director_fees','deferred_income','long_term_incentive','from_poi_to_this_person'] 

	### Load the dictionary containing the dataset
	with open("final_project_dataset.pkl", "r") as data_file:
	    data_dict = pickle.load(data_file)

	### Task 2: Remove outliers
	del data_dict['TOTAL']   

	### Extract features and labels from dataset for local testing
	data = featureFormat(data_dict, features_list, sort_keys = True)
	labels, features = targetFeatureSplit(data)

	from sklearn.feature_selection import SelectKBest

	### Use SlectKBest to figure out featuers
	clf = SelectKBest(k=4)
	new_features = clf.fit_transform(features,labels)
	params = clf.get_params()

	i=0
	featureImportance = []
	for item in clf.scores_:
	    featureImportance.append((item,features_list[i+1]))
	    i=i+1

	featureImportance=sorted(featureImportance, reverse=True)
	for item in featureImportance:
	     print "{0} , {1:4.2f}%".format(item[1],item[0])

def poiIdFunctions(features_list,features_to_scale = [], classifier = "GaussianNB", max_depth = None, min_samples_split = 2, kernel = 'rbf', C = 1000,
	degree = 3, max_features = 'auto', n_estimators = 10, pca = False,n_components=50, doGrid = False):
	### Load the dictionary containing the dataset
	with open("final_project_dataset.pkl", "r") as data_file:
		data_dict = pickle.load(data_file)

	### Task 2: Remove outliers
	del data_dict['TOTAL'] ##Only outlier remover for now
	### Task 3: Create new feature(s)
	data_dict = getFractionToAndFromPoi(data_dict)
	for feature in features_to_scale:
		getScaledFeature(data_dict,feature)
		new_feature_name = "scaled_"+feature
		features_list = [new_feature_name if x==feature else x for x in features_list]
	### Store to my_dataset for easy export below.
	my_dataset = data_dict
	### Extract features and labels from dataset for local testing
	data = featureFormat(my_dataset, features_list, sort_keys = True)
	labels, features = targetFeatureSplit(data)

	from sklearn.cross_validation import train_test_split
	features_train, features_test, labels_train, labels_test = \
	train_test_split(features, labels, test_size=0.3, random_state=42)
	from sklearn.grid_search import GridSearchCV
	if pca == True:
		from sklearn.decomposition import RandomizedPCA
		from sklearn.pipeline import Pipeline
		if classifier == "GaussianNB":
			from sklearn.naive_bayes import GaussianNB
			clf = Pipeline([('pca', RandomizedPCA(n_components = n_components, whiten=True)),
			 ('gaussian', GaussianNB())])
			clf = clf.fit(features_train,labels_train)
			predictions = clf.predict(features_test)
		elif classifier == "DecisionTreeClassifier":
			from sklearn import tree
			clf = Pipeline([('pca', RandomizedPCA(n_components = n_components, whiten=True)),
			 ('decisiontree', tree.DecisionTreeClassifier(min_samples_split = min_samples_split, max_depth = max_depth))])
			clf = clf.fit(features_train,labels_train)
			predictions = clf.predict(features_test)
		elif classifier == "SVC":
			from sklearn.svm import SVC
			clf = Pipeline([('pca', RandomizedPCA(n_components = n_components, whiten=True)),
			 ('svm', SVC(kernel=kernel,C=C, degree = degree))])
			clf = clf.fit(features_train,labels_train)
			predictions = clf.predict(features_test)
		elif classifier == "RandomForestClassifier":
			from sklearn.ensemble import RandomForestClassifier
			clf = Pipeline([('pca', RandomizedPCA(n_components = n_components, whiten=True)),
			 ('randomforest', RandomForestClassifier(n_estimators = n_estimators,min_samples_split = min_samples_split,max_depth=max_depth, max_features = max_features))])
			clf = clf.fit(features_train,labels_train)
			predictions = clf.predict(features_test)
		else:
			print "Enter a real classifier you filthy animal"

	else:
		if classifier == "GaussianNB":
			from sklearn.naive_bayes import GaussianNB
			clf = GaussianNB()
			clf.fit(features_train,labels_train)
			predictions = clf.predict(features_test)
		elif classifier == "DecisionTreeClassifier":
			from sklearn import tree
			if doGrid == True:
				param_grid = {
				        'criterion':['gini','entropy'],
				         'splitter':['best','random'],
				         'min_samples_split':[2,3,5,10],
				         'min_samples_leaf':[1,2,3,5,10,20,30],
				         'max_features':[1,2,3,4]
				          }
				# for sklearn version 0.16 or prior, the class_weight parameter value is 'auto'
				clf = GridSearchCV(tree.DecisionTreeClassifier(), param_grid,scoring="f1")
				clf = clf.fit(features_train,labels_train)
				print "Best estimator found by grid search for DecisionTree:"
				print clf.best_estimator_
				print clf.best_score_
				clf = tree.DecisionTreeClassifier(**clf.best_params_)
				clf = clf.fit(features_train,labels_train)
				predictions = clf.predict(features_test)
			else:
				clf = tree.DecisionTreeClassifier(min_samples_split = min_samples_split, max_depth = max_depth)
				clf = clf.fit(features_train,labels_train)
				predictions = clf.predict(features_test)
		elif classifier == "SVC":
			from sklearn.svm import SVC
			clf = SVC(kernel=kernel,C=C, degree = degree)
			clf = clf.fit(features_train,labels_train)
			predictions = clf.predict(features_test)
		elif classifier == "RandomForestClassifier":
			from sklearn.ensemble import RandomForestClassifier
			clf = RandomForestClassifier(n_estimators = n_estimators,min_samples_split = min_samples_split,max_depth=max_depth, max_features = max_features)
			clf = clf.fit(features_train,labels_train)
			predictions = clf.predict(features_test)
		else:
			print "Enter a real classifier you filthy animal"

	from tester import test_classifier

	test_classifier(clf,my_dataset,features_list)
