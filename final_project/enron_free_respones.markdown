
###1. Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]

The goal of this project is to build an algorithm to identify Enron employees who may have committed fraud based on the provided Enron financial dataset.  Machine learning is useful by helping us investigate patterns of the data and making predictions of classification. The classification used in the provided dataset was whether or not an Enron employee was a Person of Interest (POI).  A POI was determined by whether or not he or she was indicted by prosecutors, or settled with prosecutors, or testified for immunity with prosectors. A POI in the provided dataset under the feature "poi" was classified as True or 1, whereas a non-POI was classified as a False or 0.  In total, the provided dataset has 146 entries for Enron employees and 21 features for each these entries, making the total number of data points equal to 3,066 (146*21).  The number of Enron employees classified as POIs is 18.  Out of the 21 features, fourteen are associated with financial data in the form of payments and stock options.  The remaining five, sans "poi," are associated with email counts to and fro POIs and the employee"s email. The dataset has missing values throughout and the missing values are labeled as datatype string "NaN."  A summary dictionary below reports the percentage of data that is "NaN," or missing.  Some features such as "loan_advances" have as high as 97% of their data missing.  Based on the lack of data on some of the features, a case could be made for reducing the dimensionality. For example, the total_stock_value and and total_payments sum up the three stock features and nine payment features, respectively. The summation features may already envelope their constituents. The summation features also have the least amount of data missing at ~14% with the exception of "poi" and "name," which is simply the names of the employees in my created data frame.  The handling of "NaN" values was accomplished by the provided function featureFormat(); the function, which inputs a dictionary and outputs an array, in the class feature_format, eliminates the row with the greatest missing values of ("NaN"), reducing the data points from downstream usage.  The missing values that are not eliminated via featureFormat() are set equal to zero for future usage downstream.  I have an example of this upon request.  Some techniques for reassigning missing values are to replace them with the mean or zero.  These techniques were not implemented since the project requirements for metrics were met.    

```
{"bonus": 43.75,
 "deferral_payments": 73.61,
 "deferred_income": 66.67,
 "director_fees": 88.89,
 "email_address": 22.92,
 "exercised_stock_options": 29.86,
 "expenses": 34.72,
 "from_messages": 40.28,
 "from_poi_to_this_person": 40.28,
 "from_this_person_to_poi": 40.28,
 "loan_advances": 97.92,
 "long_term_incentive": 54.86,
 "name": 0.0,
 "other": 36.81,
 "poi": 0.0,
 "restricted_stock": 24.31,
 "restricted_stock_deferred": 88.19,
 "salary": 34.72,
 "shared_receipt_with_poi": 40.28,
 "to_messages": 40.28,
 "total_payments": 14.58,
 "total_stock_value": 13.19}
```

Outliers were determined by 2-D visualization, descriptive statistics, and data frame exploration of the features. The 2-D visualizations depicted outliers that could produce leverage on regressions; the standard deviations of the features were greater than all other descriptive statistics, signaling enormous range and variance; and data frame exploration revealed an agency, which is not an employee and thus outside of the scope of the goal.  Some notable outliers were "TOTAL" and "THE TRAVEL AGENCY IN THE PARK," both of which are not employees of Enron.  These outliers were popped from the data dictionary.  As was stated before, featureFormat() removes rows with the least amount of data, eliminating them from downstream use. 

###2. What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “properly scale features”, “intelligently select feature”]

I engineered four features: "ratio_stock_to_pay", "ratio_from_poi", "ratio_to_poi", and "ratio_cc_poi".  The feature "ratio_stock_to_pay" is the ratio of "total_stock_value" over "total_payments."  The rationale was to compress the summation of financial features into one feature to reduce dimensionality.  The feature "ratio_from_poi" is the ratio of "from_poi_to_this_person" over "to_messages."  The rationale was to compress the email-count-to features into one feature to reduce dimensionality.  The feature "ratio_to_poi" is the ratio of "from_this_person_to_poi" over "from_messages."  The rationale was to compress the email-count-from features into one feature to reduce dimensionality.  The feature "ratio_cc_poi" is the ratio of "shared_receipt_with_poi" over "to_messages."  The rationale was to reduce dimensionality and examine the influence of this feature on the classification.  With no algorithm tuning, the initial effect of these new features as my selected features produced values above 0.3 for both recall and precision, using a Gaussian Naive Bayes classifier and the cross-validation provided in `tester.py`.  

The selection processes used for feature selection was the following: hand-selected, recursive, and optimized via GridSearchCV and Pipeline that uses univariate selection per the Select-K-Best algorithm. I performed all of them to get experience with all of them.   

The hand-selected features were the ones mentioned in the previous paragraph: "ratio_stock_to_pay", "ratio_from_poi", "ratio_to_poi", and "ratio_cc_poi." The results were not within the minimum precision and recall requirements. The results of the `tester.py` are presented below:

```
features_list: ["poi", "ratio_stock_to_pay", "ratio_from_poi", "ratio_to_poi", "ratio_cc_poi"]

DecisionTreeClassifier(class_weight=None, criterion="gini", max_depth=None,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter="best")
Accuracy: 0.79392       Precision: 0.30808      Recall: 0.27250 F1: 0.28920     F2: 0.27894
Total predictions: 13000
True positives:  545    False positives: 1224   False negatives: 1455   True negatives: 9776

GaussianNB()
Accuracy: 0.69977       Precision: 0.28870      Recall: 0.65000 F1: 0.39982     F2: 0.51988
Total predictions: 13000
True positives: 1300    False positives: 3203   False negatives:  700   True negatives: 7797

KNeighborsClassifier(algorithm="auto", leaf_size=30, metric="minkowski",
           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
           weights="uniform")
Accuracy: 0.82123       Precision: 0.06216      Recall: 0.01150 F1: 0.01941     F2: 0.01374
Total predictions: 13000
True positives:   23    False positives:  347   False negatives: 1977   True negatives: 10653

AdaBoostClassifier(algorithm="SAMME.R", base_estimator=None,
          learning_rate=1.0, n_estimators=50, random_state=None)
Accuracy: 0.79969       Precision: 0.29343      Recall: 0.21450 F1: 0.24783     F2: 0.22670
Total predictions: 13000
True positives:  429    False positives: 1033   False negatives: 1571   True negatives: 9967

RandomForestClassifier(bootstrap=True, class_weight=None, criterion="gini",
            max_depth=None, max_features="auto", max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
Accuracy: 0.81585       Precision: 0.31450      Recall: 0.16700 F1: 0.21816     F2: 0.18429
Total predictions: 13000
True positives:  334    False positives:  728   False negatives: 1666   True negatives: 10272
```

The recursive feature selection used the function REFCV(). The RFECV() function uses cv = StratifiedShuffleSplit() for cross-validation. The seed for the StratifiedShuffleSplit() was set to 15, and not 42 as is the seed in `tester.py`.  The number of folds was 10.  The different seed was used to mitigate the use of the same data for cross-validation in `tester.py`.  The classifiers used were Decision Tree (DTC), AdaBoost (ABC), and Random Forrest (RFC).  The best performing classifier was ABC.  The run times of the ABC, DTC, and RFC in RFECV() were ~35.69s, ~0.76s, ~8.7s, respectively.  The performance and feature importances of ABC were:

```
 feature_list: ["poi", 
        "to_messages", 
        "bonus",
              "total_stock_value",
              "expenses",
              "from_poi_to_this_person",
              "ratio_to_poi",
              "from_this_person_to_poi",
              "deferred_income",
              "ratio_stock_to_pay",
              "salary",
              "ratio_from_poi",
              "total_payments",
              "exercised_stock_options",
              "restricted_stock",
              "shared_receipt_with_poi",
              "other",
              "ratio_cc_poi"]

 AdaBoostClassifier(algorithm="SAMME.R", base_estimator=None,
          learning_rate=1.0, n_estimators=50, random_state=None)
Accuracy: 0.86493       Precision: 0.49131      Recall: 0.36750 F1: 0.42048     F2: 0.38701
Total predictions: 15000
True positives:  735    False positives:  761   False negatives: 1265   True negatives: 12239

            full_feature_list  rfecv.ranking_
0                 to_messages               1
1           deferral_payments               5
2                       bonus               1
3           total_stock_value               1
4                    expenses               1
5     from_poi_to_this_person               1
6                ratio_to_poi               1
7     from_this_person_to_poi               1
8             deferred_income               1
9         long_term_incentive               3
10         ratio_stock_to_pay               1
11                     salary               1
12             ratio_from_poi               1
13             total_payments               1
14    exercised_stock_options               1
15           restricted_stock               1
16  restricted_stock_deferred               2
17    shared_receipt_with_poi               1
18              loan_advances               4
19              from_messages               6
20                      other               1
21               ratio_cc_poi               1
22              director_fees               7
```

Herein the ranking in the rfecv.ranking_ column is from one to the 21. Ties are equally important.  Features with a ranking of one are the highest ranked and kept for subsequent use.  

I used a Pipeline and GridSearch to try to optimize feature selection for recall and precision, or F1 (2*precision*recall)/(precision + recall).  A Pipelines consisted of a minMaxScaler, Principal Component Analysis (PCA), Select-K-Best (SKB), and a classifier:  Support Vector Machines (SVM).  Other Pipelines consisted of SKB and then the classifier:  GNB, DTC, and ABC.  Only SVM needed scaling since those algorithms maximize margin to perform classification, and trade-off one dimension to another, so data with orders of magnitude in difference will have a considerable effect.  In addition, performing PCA is assisted by scaling.  Here too, orders of magnitude may mask true variance and perturb transformations to incorrect axises.   GNB roughly uses probabilities of feature values to make predictions and assumes independence of the features.  The decision tree algorithms DTC ABC, and RFC partition based on maximizing information gain.  GNB, DTC, ABC, RFC are not impacted by scaling.  

GNB results are below:

```
feature_list = ["bonus", "total_stock_value", "ratio_to_poi", "deferred_income", "long_term_incentive", "salary", "exercised_stock_options"]
                 feature_list     scores
0                 to_messages   1.646341
1           deferral_payments   0.224611
2                       bonus  20.792252
3           total_stock_value  24.182899
4                    expenses   6.094173
5     from_poi_to_this_person   5.243450
6                ratio_to_poi  16.409713
7     from_this_person_to_poi   2.382612
8             deferred_income  11.458477
9         long_term_incentive   9.922186
10         ratio_stock_to_pay   0.022819
11                     salary  18.289684
12             ratio_from_poi   3.128092
13             total_payments   8.772778
14    exercised_stock_options  24.815080
15           restricted_stock   9.212811
16  restricted_stock_deferred   0.065500
17    shared_receipt_with_poi   8.589421
18              loan_advances   7.184056
19              from_messages   0.169701
20                      other   4.187478
21               ratio_cc_poi   9.101269
22              director_fees   2.126328
---------------------------------------------------------------
        skb__k: 7
GridSearch time:
0.45
-----------------------------------------------------------------------
Pipeline(steps=[("skb", SelectKBest(k=7, score_func=<function f_classif at 0x10ee276e0>)), ("gnb", GaussianNB())])
Accuracy: 0.84307       Precision: 0.39527      Recall: 0.33400 F1: 0.36206     F2: 0.34469
Total predictions: 15000
True positives:  668    False positives: 1022   False negatives: 1332   True negatives: 11978

test_classifier time:
2.27
total time: 2.72
```

DTC results are below: 

```
feature_list = ["bonus", "total_stock_value", "exercised_stock_options"]
                 feature_list     scores
0                 to_messages   1.646341
1           deferral_payments   0.224611
2                       bonus  20.792252
3           total_stock_value  24.182899
4                    expenses   6.094173
5     from_poi_to_this_person   5.243450
6                ratio_to_poi  16.409713
7     from_this_person_to_poi   2.382612
8             deferred_income  11.458477
9         long_term_incentive   9.922186
10         ratio_stock_to_pay   0.022819
11                     salary  18.289684
12             ratio_from_poi   3.128092
13             total_payments   8.772778
14    exercised_stock_options  24.815080
15           restricted_stock   9.212811
16  restricted_stock_deferred   0.065500
17    shared_receipt_with_poi   8.589421
18              loan_advances   7.184056
19              from_messages   0.169701
20                      other   4.187478
21               ratio_cc_poi   9.101269
22              director_fees   2.126328
---------------------------------------------------------------
        dtc__min_samples_split: 1
        skb__k: 3
GridSearch time:
0.64
-----------------------------------------------------------------------
Pipeline(steps=[("skb", SelectKBest(k=3, score_func=<function f_classif at 0x10ee276e0>)), ("dtc", DecisionTreeClassifier(class_weight=None, criterion="gini", max_depth=None,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=1, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter="best"))])
Accuracy: 0.80393       Precision: 0.28288      Recall: 0.30650 F1: 0.29422     F2: 0.30147
Total predictions: 15000
True positives:  613    False positives: 1554   False negatives: 1387   True negatives: 11446

test_classifier time:
1.88
total time: 2.52
```

The results for ABC are: 

```
["bonus", "total_stock_value", "exercised_stock_options"]
                 feature_list     scores
0                 to_messages   1.646341
1           deferral_payments   0.224611
2                       bonus  20.792252
3           total_stock_value  24.182899
4                    expenses   6.094173
5     from_poi_to_this_person   5.243450
6                ratio_to_poi  16.409713
7     from_this_person_to_poi   2.382612
8             deferred_income  11.458477
9         long_term_incentive   9.922186
10         ratio_stock_to_pay   0.022819
11                     salary  18.289684
12             ratio_from_poi   3.128092
13             total_payments   8.772778
14    exercised_stock_options  24.815080
15           restricted_stock   9.212811
16  restricted_stock_deferred   0.065500
17    shared_receipt_with_poi   8.589421
18              loan_advances   7.184056
19              from_messages   0.169701
20                      other   4.187478
21               ratio_cc_poi   9.101269
22              director_fees   2.126328
---------------------------------------------------------------
        abc__learning_rate: 1
        abc__n_estimators: 50
        skb__k: 3
GridSearch time:
39.4
-----------------------------------------------------------------------
Pipeline(steps=[("skb", SelectKBest(k=3, score_func=<function f_classif at 0x10ee276e0>)), ("abc", AdaBoostClassifier(algorithm="SAMME.R", base_estimator=None, learning_rate=1,
          n_estimators=50, random_state=None))])
Accuracy: 0.81613       Precision: 0.28392      Recall: 0.24900 F1: 0.26532     F2: 0.25528
Total predictions: 15000
True positives:  498    False positives: 1256   False negatives: 1502   True negatives: 11744

test_classifier time:
147.38
total time: 186.78
```
The results for SVM are: 

```
features_list = ["exercised_stock_options"]
                 feature_list     scores
0                 to_messages   1.646341
1           deferral_payments   0.224611
2                       bonus  20.792252
3           total_stock_value  24.182899
4                    expenses   6.094173
5     from_poi_to_this_person   5.243450
6                ratio_to_poi  16.409713
7     from_this_person_to_poi   2.382612
8             deferred_income  11.458477
9         long_term_incentive   9.922186
10         ratio_stock_to_pay   0.022819
11                     salary  18.289684
12             ratio_from_poi   3.128092
13             total_payments   8.772778
14    exercised_stock_options  24.815080
15           restricted_stock   9.212811
16  restricted_stock_deferred   0.065500
17    shared_receipt_with_poi   8.589421
18              loan_advances   7.184056
19              from_messages   0.169701
20                      other   4.187478
21               ratio_cc_poi   9.101269
22              director_fees   2.126328
---------------------------------------------------------------
        features__pca__n_components: 2
        features__pca__whiten: True
        features__univ_select__k: 1
        svm__C: 1
        svm__gamma: 0.9
        svm__kernel: "sigmoid"
GridSearch time:
8.91
-----------------------------------------------------------------------
Pipeline(steps=[("features", FeatureUnion(n_jobs=1,
       transformer_list=[("scale", MinMaxScaler(copy=True, feature_range=(0, 1))), ("pca", PCA(copy=True, n_components=2, whiten=True)), ("univ_select", SelectKBest(k=1, score_func=<function f_classif at 0x10ee276e0>))],
       transformer_weights=None)), (...,
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False))])
Accuracy: 0.86780       Precision: 0.51220      Recall: 0.17850 F1: 0.26474     F2: 0.20524
Total predictions: 15000
True positives:  357    False positives:  340   False negatives: 1643   True negatives: 12660

test_classifier time:
4.8
total time: 13.71
```

The features scores done by SKB were consistent due to method of analysis: ANOVA.   The features selected were due to optimization done by GridSearch using the selected algorithm. 

###3. What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”] What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric item: “tune the algorithm”]

I selected the GNB Classifier for its simplicity.  The GNB did not need scaling; it works well with or without PCA; it required no tuning as parameters for the algorithm do not exist; it"s runtimes were the fastest.  The other algorithms that I tried are the following algorithms: SVM, DTC, ABC, and RFC.  The results of performance are in section 2.  In general, accuracy scores from highest to least -- while performing GridSearch -- were:  SVM, GNB, DTC, and ABC; precision scores were: SVM, GNB, ABC, and DTC; recall scores were:  ABC, GNB, DTC, and ABC; the runtimes were:  ABC, SVM, DTC, and GNB.  These results vary from the recursive algorithm RFECV().  The difference perhaps is attributed to the way the algorithms seeks the optimized solution.  RFECV() does not use SKB, which calculates feature selection via ANOVA, that is used by GridSearch via Pipeline.  RFECV() uses different combinations of features and is forced to eliminate the least contributing feature each iteration per its scoring.  These differences could be responsible for the disparate results.        

Parameter tuning means to manipulate parameters of the algorithm to best fit the data based on a desired metric such as precision or recall.  Tuning the parameters too well may lead to models that predict training data very well but are unable to predict on unseen data or testing data.  Tuning parameters badly leads to misclassification and over-complication of models . This can lead to increased runtime and coarse generality which could not assist prediction.  I performed parameter tuning through the GridSearch for multiple classifiers. The general procedure is parametric. You change the inputs of the parameters and analyze the results.  You reselect different inputs for your parameters and continue to analyze results until you have a desired or copacetic result.  Typically, you provide coarse inputs first and then finer inputs as your performance goes up.  This way you can hone-in on a finely tuned algorithm.  Too much tune-age results in test dataset leakage. The held out test dataset for validation may leak into the train dataset with finely tuned parameters.          


###4. What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric item: “validation strategy”] Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”

Validation is evaluating your machine learning algorithm on a dataset that has been withheld from the training dataset.  The classic mistake is overfitting or not partitioning the data correctly into three distinct training, validation, and testing sets.  Overfit models predict overly excellent.  The data has already been seen my the model so it knows the correct answer.  The result is a model that is unable to predict correctly on unseen data.  

I cross-validated my analysis with the `tester.py` function.  "tester.py" uses StratifiedShuffleSplit() (SSS) to partition the data into 128 training points and 15 testing points a 1000 times.  SSS uses the the same distribution of the original dataset in its training and testing sets. The reason for using SSS is that our original dataset is unbalanced and other cross-validation testing and training splits may produce data splits that may not have any POIs in them, making them poor representatives of the dataset.  The original dataset had 18 POIs and 125 non-POIs, meaning that roughly 16 POIs are used in the training set and roughly 2 are used in testing set.  The combinatorics supports 1000 random shuffle splits with these ratios. For the optimizing and recursive procedures, I used SSS as well with a fold of 10 and a different random number starting point in the attempt to mitigate leakage.  

The metrics sought were precision and recall.  Precision is the ability of the classifier not to label as positive a sample that is negative. Recall is the ability of the classifier to find all the positive samples.  In laymen terms and applicable to this project,  precision is likelihood that the algorithm has predicted a POI correctly, whereas recall is likelihood that a POI has correctly been predicted to be a POI.  A ~30 percent likelihood of both recall and precision is not the most stellar of predictions, but the classification is not critical as in medicine application, so a ~30 percent likelihood could be acceptable as a first-cut analysis tool for later and subsequent traditional investigative techniques. 

> Written with [StackEdit](https://stackedit.io/).