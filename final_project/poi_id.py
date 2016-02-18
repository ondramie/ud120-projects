#!/usr/bin/python

import sys
import pickle
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time

sys.path.append("../tools/")

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFECV
from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from pprint import pprint

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
### for the full_report
full_report = False

###############################################################################
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','ratio_stock_to_pay', 'ratio_from_poi', \
                 'ratio_to_poi', 'ratio_cc_poi'] 

if full_report:
    print "features_list:", features_list


###############################################################################
### Task 2: Remove some outliers
data_dict.pop("TOTAL",0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK",0)

#### remove some employees with very high stock_to_pay_ratios?
if False:
    data_dict.pop("HAUG DAVID L",0)
    data_dict.pop("HIRKO JOSEPH",0)
    data_dict.pop("REDMOND BRIAN L",0)
    data_dict.pop("RICE KENNETH D",0)

#### function examines the descriptive statistics of the features and 
#### the percentage of NaNs in the features
def examineData_Dict(enron_data):
    
    ## creation of dataframe 
    df_enron_data = pd.DataFrame(enron_data).T
    print "Total No. of datapoints:", df_enron_data.shape
    
    ## changing index of dataframe
    df_enron_data.reset_index(inplace = True)
    df_enron_data.rename(columns = {'index':'name'}, inplace = True)
    
    
    print "No. of employees:", 
    num_employees = len(df_enron_data.index)
    print num_employees
    print "No. of features:", len(df_enron_data.columns)
    
    ## no. of pois    
    df_POIs = df_enron_data[df_enron_data.poi == 1]
    print "No. of POIs", len(df_POIs)
    
    labels = df_enron_data.poi.tolist() 
    features = df_enron_data.columns.tolist()
    
    ## replace string "NaN" with np.nan
    df_enron_data_npnan = df_enron_data.replace(to_replace = "NaN",
                                                value = np.nan)
                                                
    numeric_feat = []
    dic_of_nans = {}
    for feature in features:
        ### removes features non-numeric
        if feature not in ["email_address","poi","name"]:
            numeric_feat.append(feature)     
        
        ### descriptive statistics of entries    
        print         
        print feature + ": "    
        pprint(df_enron_data_npnan[feature].describe())
    
        ### calculates percentage of NaNs per feature
        nan_in_column = []
        for item in df_enron_data[feature]: 
            if item == "NaN":            
                nan_in_column.append(1)
            
            dic_of_nans.update({ feature : 
                round(float(sum(nan_in_column)/float(num_employees))*100,2)})
                
    print            
    print "the percentage of missing data (NaNs) per feature:"
    pprint(dic_of_nans)
    
if full_report:
    examineData_Dict(data_dict)

###############################################################################
### Task 3: Create new feature(s)
#### features of interest to create new features
foi_list = ['total_payments','total_stock_value','from_this_person_to_poi', \
            'from_poi_to_this_person', 'to_messages', 'from_messages', \
            'shared_receipt_with_poi']

#### function to create new features; computes ratio 
def computeRatio(num,denom):
    if num != 'NaN' and denom != 'NaN':
        fraction = float(num)/float(denom)
    else:
        fraction = 'NaN'
    return fraction  

foi_data = []
#### new features added to data_dict     
for name in data_dict:
    employee = data_dict[name]
    employee['ratio_stock_to_pay'] = computeRatio(employee[foi_list[1]], employee[foi_list[0]])
    employee['ratio_from_poi'] = computeRatio(employee[foi_list[3]], employee[foi_list[4]])                                               
    employee['ratio_to_poi'] = computeRatio(employee[foi_list[2]], employee[foi_list[5]])                                               
    employee['ratio_cc_poi'] = computeRatio(employee[foi_list[6]], employee[foi_list[4]])
    
    ##### partially Invesitigate some potential outliers    
    if full_report:
        print 
        print "employees with high stock to pay ratios:"
        if employee['ratio_stock_to_pay'] > 40.0 and employee['ratio_stock_to_pay'] != 'NaN':
            print name
            pprint(employee)
            print
        
### Store to my_dataset for easy export below.
my_dataset = data_dict

#### full list of features for recursive and optimized feature selection
full_features_list = []
for entry in my_dataset.values():
    full_features_list = list(set(entry))
full_features_list.insert(0, full_features_list.pop(full_features_list.index("poi")))
full_features_list.pop(full_features_list.index("email_address"))

#### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#### Extract full features and full labels for local testing
data1 = featureFormat(my_dataset, full_features_list, sort_keys = True)
fullset_labels, fullset_features = targetFeatureSplit(data1)

#### some plots of selected features
if full_report:
    plt.scatter(np.array(features)[:,0], np.array(features)[:,2], s = 30, 
                c = labels, cmap = plt.cm.bwr)
    plt.xlabel(features_list[1])
    plt.ylabel(features_list[3])
    plt.show()
    
    plt.scatter(np.array(features)[:,0], np.array(features)[:,1], s = 30, 
                c = labels, cmap = plt.cm.bwr)
    plt.xlabel(features_list[1])
    plt.ylabel(features_list[2])
    plt.show()        
    
    plt.scatter(np.array(features)[:,0], np.array(features)[:,3], s = 30, 
                c = labels, cmap = plt.cm.bwr)
    plt.xlabel(features_list[1])
    plt.ylabel(features_list[4])
    plt.show() 

# classifiers under consideration
svc = SVC()
dtc = DecisionTreeClassifier()
svm = SVC()
gnc = GaussianNB()
knn = KNeighborsClassifier()
abc = AdaBoostClassifier()
rfc = RandomForestClassifier()

t000 = time()

#### ---------------- recursive feature_selection ----------------------------- 

if full_report:
    dic_rfecv_alg = []
    for clf in [dtc,abc,rfc]:    
        t000 = time()
        
        #### rfecv object
        rfecv = RFECV(estimator=clf, 
                      step=1, 
                      cv=StratifiedShuffleSplit(fullset_labels,
                                                10, 
                                                random_state = 15),
                      scoring='f1')
        
        #### fit on fullset of features          
        rfecv_clf = rfecv.fit(fullset_features, fullset_labels)    
        rfecv_best = rfecv_clf.estimator_
        
        rfecv_features_list = []
        for (x,y) in zip(full_features_list[1:],rfecv.support_):
            if y: 
                rfecv_features_list.append(x)
        
        #### populate list of rfecv times
        dic_rfecv_alg.append({"algorithm" : clf,
                            "run_time" : round(time()-t000,2),
                            "rfecv_features_list": rfecv_features_list})   
        
        #### Plot number of features VS. cross-validation scores
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        plt.show()        
        
        print 
        #### second cross-validation per test_classifier
        rfecv_features_list.insert(0,'poi')
        test_classifier(rfecv_best, my_dataset, rfecv_features_list)                        
        
        pprint(pd.DataFrame({"rfecv.ranking_" : rfecv.ranking_,
                            "full_feature_list" : full_features_list[1:]}))
    
    pprint(dic_rfecv_alg)

####-------------- end of recursive feature_selection -------------------------


###############################################################################
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

if True:
    # initial components 
    scaler = MinMaxScaler()
    pca = PCA()
    skb = SelectKBest(score_func=f_classif)
    
    # Build estimator from PCA and Univariate selection:
    select = FeatureUnion([("scale", scaler),
                                      ("pca", pca), 
                                      ("univ_select", skb)])
    
    # Create test and train
    sss = StratifiedShuffleSplit(fullset_labels,
                                 10, 
                                 #test_size = .10,
                                 random_state = 0)
    t00 = time() 
    
    print "sss time:", time()-t00
    
    #features_train, features_test, labels_train, labels_test = \
    tts = train_test_split(fullset_features, fullset_labels, test_size=0.4, random_state=0)
    skf = StratifiedKFold(fullset_labels, 10)
        
    t0 = time()
    #print svm.__dict__['_impl']    
    
    #### pipelines 
    pipeline_svm = Pipeline([("features", select), 
                         ("svm", svm)])
            
    pipeline_dtc = Pipeline([("skb", skb), ("dtc", dtc)])                     
    
    pipeline_gnc = Pipeline([("skb", skb),("gnc", gnc)])
    
    pipeline_abc = Pipeline([("skb", skb), ("abc", abc)])
    
    pipeline_rfc = Pipeline([("skb", skb), ("rfc", rfc)])                                          
                         
    pipeline_knn = Pipeline([("skb", skb),
    #("features", select), 
                         ("knn", knn)])
   
   ### gridsearch                   
    param_grid_svm = dict(features__pca__n_components=range(1,3),
                      features__pca__whiten = [False,True], 
                      features__univ_select__k=range(1,2),
                      svm__kernel = ['rbf','sigmoid'],
                      svm__C=[0.01,0.1,1],
                      svm__gamma = [0.1,0.9])
                      
    param_grid_dtc = dict(skb__k = range(1,4),
                          dtc__min_samples_split = [1,10,100])

    param_grid_gnc = dict(skb__k = range(1,10))

    param_grid_abc = dict(skb__k = range(1,4),
                      abc__learning_rate=[1, 10, 100],
                      abc__n_estimators = [1,50,100])
    
    param_grid_knn = dict(#features__pca__n_components=range(1,3),
                      #features__pca__whiten = [False,True], 
                      #features__univ_select__k=range(1,2),
                      knn__n_neighbors = range(2,3,6))
                      
    grid_search = GridSearchCV(pipeline_gnc, 
                               param_grid=param_grid_gnc, 
                               verbose=1,
                               scoring = "f1",
                               error_score = 0,
                               cv = sss)
                               #n_jobs = -1)
                                                          
    grid_search.fit(np.array(fullset_features), np.array(fullset_labels))    
    print "-------------------------------------------------------------------"
    clf_best = grid_search.best_estimator_
    print "best pipeline:", clf_best
    best_parameters = grid_search.best_estimator_.get_params()
    print 
    if grid_search.param_grid in [param_grid_svm]:
        skbs = clf_best.named_steps['features'].transformer_list[2][1]
        print skbs.scores_
       
        feats = []        
        list_scores = []            
        for x,y in zip(skbs.get_support(),
                         full_features_list[1:]):        
            if x:
                feats.append(y)
        
        for x,y in zip(full_features_list[1:], skbs.scores_): 
            list_scores.append({'feature_list' : x, "scores" : y})
        
        print feats
        print pd.DataFrame(list_scores)        
        print "---------------------------------------------------------------"
        for param_name in sorted(grid_search.param_grid.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))
       
    else:
        print clf_best.named_steps['skb']    
        print clf_best.named_steps['skb'].scores_    
    
        feats = []        
        list_scores = []            
        for x,y in zip(clf_best.named_steps['skb'].get_support(),
                         full_features_list[1:]):        
            if x:
                feats.append(y)
        
        for x,y in zip(full_features_list[1:], clf_best.named_steps['skb'].scores_): 
            list_scores.append({'feature_list' : x, "scores" : y})
        
        print feats
        print pd.DataFrame(list_scores)        
        print "---------------------------------------------------------------"
        for param_name in sorted(grid_search.param_grid.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))

    print "GridSearch time:" 
    time1 = round(time()-t0,2)
    print time1
    print "test_classifier time:" 
    time2 = round(time()-t1, 2)
    test_classifier(clf_best, my_dataset, full_features_list)
    print time2
    print "total time:", time2+time1
print "-----------------------------------------------------------------------"
############################################################################### 
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import train_test_split
#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

t1 = time()
if full_report:
    for clf in [ dtc, gnc, knn, abc, rfc ]:
       test_classifier(clf, my_dataset, features_list)

dump_classifier_and_data(clf_best, my_dataset, full_features_list)