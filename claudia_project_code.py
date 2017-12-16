#Header with name, class, etc. and explanation of what the code does

import pandas as pd
import numpy as np
from sklearn import naive_bayes
from sklearn.svm import SVC
from sklearn import ensemble
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


#Converts non_numeric values within the dataset into numeric indicators
def convert_non_numeric_data(df):
    columns = df.columns.values
    
    race_codes = {}
    
    for column in columns:
        text_digit_vals = {}

        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_content = df[column].values.tolist()
            unique_elements = set(column_content)
            x = 0

            for unique in unique_elements:
                if unique not in text_digit_vals:
                
                    text_digit_vals[unique] = x
                
                    if column == 'race':
                        race_codes[x] = unique
                
                x += 1

            df[column] = list(map(convert_to_int, df[column]))
    
    return df, race_codes


def calc_metrics(train_labels, test_labels, predict_test_labels, predict_train_labels):
    train_acc = metrics.accuracy_score(train_labels, predict_train_labels)
    test_acc = metrics.accuracy_score(test_labels, predict_test_labels)

    
    train_prec = metrics.precision_score(train_labels, predict_train_labels, average='micro')
    test_prec = metrics.precision_score(test_labels, predict_test_labels, average='micro')
    
    confusion_matrix = metrics.confusion_matrix(test_labels, predict_test_labels)
    
    #true positives
    tp = np.diag(confusion_matrix)
    
    #false positives
    fp = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    
    #false negatives
    fn = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    
    #true negatives
    tn = confusion_matrix.sum() - (fp + fn + tp)

    
    return test_acc, train_acc, train_prec, test_prec, tp, fp, fn, tn


#Split dataframe into necessary subsections
def split_labels(df):
    
    data = df.drop(columns=['is_recid', 'is_violent_recid'])
    data_limit = df.drop(columns=['is_recid', 'is_violent_recid', 'race'])
    
    labels = df[['is_recid']]
    labels = labels.as_matrix()
    labels = labels.ravel()
    
    alt_labels = df[['is_violent_recid']]
    alt_labels = alt_labels.as_matrix()
    alt_labels = alt_labels.ravel()
    
    return data, data_limit, labels, alt_labels



def model_training(df, race_codes):

    #
    # DATA PROCESSING
    #
    #shuffle the rows of the dataset
    df = df.sample(frac=1).reset_index(drop=True)
    rows = len(df)

    #using the reset indices, split the dataset into training and test sets
    offset_1 = 7000
    offset_2 = 9000
    train_df = df.iloc[:offset_1]
    test_df = df.iloc[offset_1 + 1:offset_2]
    dev_df = df.iloc[offset_2 + 1:]

    #split dataframe into subsections
    train_data, train_data_limit, train_labels, train_alt_labels = split_labels(train_df)
    test_data, test_data_limit, test_labels, test_alt_labels = split_labels(test_df)
    
    #determine the numeric values that correspond to race groups in the converted dataframe
    aa_index = 0
    white_index = 0
    
    for x in race_codes:
        if race_codes[x] == 'African-American':
            aa_index = x
        if race_codes[x] == 'Caucasian':
            white_index = x
    
    #split dataframe by race into subsections (make this make more sense)
    aa_df = df[df.race == aa_index]
    aa_df = aa_df.reset_index()
    aa_df = aa_df.drop(columns=['index'])
    
    white_df = df[df.race == white_index]
    white_df = white_df.reset_index()
    white_df = white_df.drop(columns=['index'])
    
    
    #separate the race specific dataframes into test, train, and dev sets
    shape1 = white_df.shape
    
    white_offset_1 = int(round(white_df.shape[0]*0.7))
    white_offset_2 = int(round(white_df.shape[0]*0.9))
    
    white_train_df = white_df.iloc[:white_offset_1]
    white_test_df = df.iloc[white_offset_1 + 1:white_offset_2]
    white_dev_df = df.iloc[white_offset_2 + 1:]
    
    aa_train_df = aa_df.iloc[:white_offset_1]
    aa_test_df = df.iloc[white_offset_1 + 1:white_offset_2]
    aa_dev_df = df.iloc[white_offset_2 + 1:shape1[1]]


    aa_train_data_limit = aa_train_df.drop(columns=['is_recid', 'is_violent_recid', 'race'])
    aa_train_labels = aa_train_df[['is_recid']]
    aa_train_alt_labels = aa_train_df[['is_violent_recid']]
    
    aa_train_labels = aa_train_labels.as_matrix()
    aa_train_labels = aa_train_labels.ravel()
    aa_train_alt_labels = aa_train_alt_labels.as_matrix()
    aa_train_alt_labels = aa_train_alt_labels.ravel()
  
  
    aa_test_data_limit = aa_test_df.drop(columns=['is_recid', 'is_violent_recid', 'race'])
    aa_test_labels = aa_test_df[['is_recid']]
    aa_test_alt_labels = aa_test_df[['is_violent_recid']]

    aa_test_labels = aa_test_labels.as_matrix()
    aa_test_labels = aa_test_labels.ravel()
    aa_test_alt_labels = aa_test_alt_labels.as_matrix()
    aa_test_alt_labels = aa_test_alt_labels.ravel()


    aa_dev_data_limit = aa_dev_df.drop(columns=['is_recid', 'is_violent_recid', 'race'])
    aa_dev_labels = aa_dev_df[['is_recid']]
    aa_dev_alt_labels = aa_dev_df[['is_violent_recid']]
   
    aa_dev_labels = aa_dev_labels.as_matrix()
    aa_dev_labels = aa_dev_labels.ravel()
    aa_dev_alt_labels = aa_dev_alt_labels.as_matrix()
    aa_dev_alt_labels = aa_dev_alt_labels.ravel()
    

    white_train_data_limit = white_train_df.drop(columns=['is_recid', 'is_violent_recid', 'race'])
    white_train_labels = white_train_df[['is_recid']]
    white_train_alt_labels = white_train_df[['is_violent_recid']]
    
    white_train_labels = white_train_labels.as_matrix()
    white_train_labels = white_train_labels.ravel()
    white_train_alt_labels = white_train_alt_labels.as_matrix()
    white_train_alt_labels = white_train_alt_labels.ravel()
    
    white_test_data_limit = white_test_df.drop(columns=['is_recid', 'is_violent_recid', 'race'])
    white_test_labels = white_test_df[['is_recid']]
    white_test_alt_labels = white_test_df[['is_violent_recid']]
    
    white_test_labels = white_test_labels.as_matrix()
    white_test_labels = white_test_labels.ravel()
    white_test_alt_labels = white_test_alt_labels.as_matrix()
    white_test_alt_labels = white_test_alt_labels.ravel()

    white_dev_data_limit = white_dev_df.drop(columns=['is_recid', 'is_violent_recid', 'race'])
    white_dev_labels = white_dev_df[['is_recid']]
    white_dev_alt_labels = white_dev_df[['is_violent_recid']]

    white_dev_labels = white_dev_labels.as_matrix()
    white_dev_labels = white_dev_labels.ravel()
    white_dev_alt_labels = white_dev_alt_labels.as_matrix()
    white_dev_alt_labels = white_dev_alt_labels.ravel()





    #
    # MODEL TRAINING
    #





    #
    # NAIVE BAYES MODEL
    #
    #train baseline Naive Bayes model with no priors
    print 'Naive Bayes'

    nb = naive_bayes.GaussianNB()
    nb.fit(train_data, train_labels)
  
    baseline_nb_train_labels = nb.predict(train_data)
    baseline_nb_test_labels = nb.predict(test_data)
    
    baseline_nb_metrics = []
    baseline_nb_metrics = calc_metrics(train_labels, test_labels, baseline_nb_test_labels, baseline_nb_train_labels)
    
    
    #NB with race feature removed
    nb.fit(train_data_limit, train_labels)
    
    rr_nb_train_labels = nb.predict(train_data_limit)
    rr_nb_test_labels = nb.predict(test_data_limit)
    
    rr_nb_metrics = []
    rr_nb_metrics = calc_metrics(train_labels, test_labels, rr_nb_test_labels, rr_nb_train_labels)


    #Race specific NB models
    nb.fit(aa_train_data_limit, aa_train_labels)
    aa_nb_train_labels = nb.predict(aa_train_data_limit)
    aa_nb_test_labels = nb.predict(aa_test_data_limit)

    aa_nb_metrics = []
    aa_nb_metrics = calc_metrics(aa_train_labels, aa_test_labels, aa_nb_test_labels, aa_nb_train_labels)


    nb.fit(white_train_data_limit, white_train_labels)
    white_nb_train_labels = nb.predict(white_train_data_limit)
    white_nb_test_labels = nb.predict(white_test_data_limit)

    white_nb_metrics = []
    white_nb_metrics = calc_metrics(white_train_labels, white_test_labels, white_nb_test_labels, white_nb_train_labels)

    rf = ensemble.RandomForestClassifier(max_depth = 3)



    #NB using alternative labels
    nb.fit(train_data_limit, train_alt_labels)
    alt_nb_train_labels = nb.predict(train_data_limit)
    alt_nb_test_labels = nb.predict(test_data_limit)

    alt_nb_metrics = []
    alt_nb_metrics = calc_metrics(train_labels, test_labels, alt_nb_test_labels, alt_nb_train_labels)



    #
    # GRADIENT BOOSTING
    #
    
    #baseline GB
    print 'Gradient Boosting'

    gb = ensemble.GradientBoostingClassifier(n_estimators = 1000, max_leaf_nodes = 4, subsample = 1.0, min_samples_split = 2,
          learning_rate = 0.01)
    gb.fit(train_data, train_labels)

    baseline_gb_train_labels = gb.predict(train_data)
    baseline_gb_test_labels = gb.predict(test_data)

    baseline_gb_metrics = []
    baseline_gb_metrics = calc_metrics(train_labels, test_labels, baseline_gb_test_labels, baseline_gb_train_labels)


    #GB with race and name features removed
    gb.fit(train_data_limit, train_labels)
    limit_gb_train_labels = gb.predict(train_data_limit)
    limit_gb_test_labels = gb.predict(test_data_limit)

    rr_gb_metrics = []
    rr_gb_metrics = calc_metrics(train_labels, test_labels, limit_gb_test_labels, limit_gb_train_labels)


    #Race specific GB models
    gb.fit(aa_train_data_limit, aa_train_labels)
    aa_gb_train_labels = gb.predict(aa_train_data_limit)
    aa_gb_test_labels = gb.predict(aa_test_data_limit)

    aa_gb_metrics = []
    aa_gb_metrics = calc_metrics(aa_train_labels, aa_test_labels, aa_gb_test_labels, aa_gb_train_labels)

    gb.fit(white_train_data_limit, white_train_labels)
    white_gb_train_labels = gb.predict(white_train_data_limit)
    white_gb_test_labels = gb.predict(white_test_data_limit)

    white_gb_metrics = []
    white_gb_metrics = calc_metrics(white_train_labels, white_test_labels, white_gb_test_labels, white_gb_train_labels)


    #GB model with alternate labels
    gb.fit(train_data_limit, train_alt_labels)
    alt_gb_train_labels = gb.predict(train_data_limit)
    alt_gb_test_labels = gb.predict(test_data_limit)

    alt_gb_metrics = []
    alt_gb_metrics = calc_metrics(train_labels, test_labels, alt_gb_test_labels, alt_gb_train_labels)






    #
    # SUPPORT VECTOR MACHIMES
    #
    print 'SVM'

    #baseline SVM
    svm = SVC()
    svm.fit(train_data, train_labels)
  
    baseline_svm_train_labels = svm.predict(train_data)
    baseline_svm_test_labels = svm.predict(test_data)

    baseline_svm_metrics = []
    baseline_svm_metrics = calc_metrics(train_labels, test_labels, baseline_svm_test_labels, baseline_svm_train_labels)


    #svm with race feature removed
    svm.fit(train_data_limit, train_labels)
    
    limit_svm_train_labels = svm.predict(train_data_limit)
    limit_svm_test_labels = svm.predict(test_data_limit)
    
    rr_svm_metrics = []
    rr_svm_metrics = calc_metrics(train_labels, test_labels, limit_svm_test_labels, limit_svm_train_labels)


    #svm model for examples involving African American individuals
    svm.fit(aa_train_data_limit, aa_train_labels)
    aa_svm_train_labels = svm.predict(aa_train_data_limit)
    aa_svm_test_labels = svm.predict(aa_test_data_limit)

    aa_svm_metrics = []
    aa_svm_metrics = calc_metrics(aa_train_labels, aa_test_labels, aa_svm_test_labels, aa_svm_train_labels)



    #svm model for examples involving white individuals
    svm.fit(white_train_data_limit, white_train_labels)
    white_svm_train_labels = svm.predict(white_train_data_limit)
    white_svm_test_labels = svm.predict(white_test_data_limit)

    white_svm_metrics = []
    white_svm_metrics = calc_metrics(white_train_labels, white_test_labels, white_svm_test_labels, white_svm_train_labels)


    #svm using alternative labels
    svm.fit(train_data_limit, train_alt_labels)
    alt_svm_train_labels = svm.predict(train_data_limit)
    alt_svm_test_labels = svm.predict(test_data_limit)

    alt_svm_metrics = []
    alt_svm_metrics = calc_metrics(train_labels, test_labels, alt_svm_test_labels, alt_svm_train_labels)
    
    


    
    
    #
    # RANDOM FOREST
    #
    print 'Random Forest'

    rf = ensemble.RandomForestClassifier(max_depth = 3)
    #baseline RF
    rf.fit(train_data, train_data)
    
    baseline_rf_train_labels = rf.predict(train_data)
    baseline_rf_test_labels = rf.predict(test_data)

    print(rf.feature_importances_)


    #RF with race and name features removed
    rf.fit(train_data_limit, train_labels)
    
    limit_rf_train_labels = rf.predict(train_data_limit)
    limit_rf_test_labels = rf.predict(test_data_limit)

    print(rf.feature_importances_)


    #Race specific RF models
    #african-american model
    rf.fit(aa_train_data_limit, aa_train_labels)

    aa_rf_train_labels = rf.predict(aa_train_data_limit)
    aa_rf_test_labels = rf.predict(aa_test_data_limit)

    print(rf.feature_importances_)

    #white model
    rf.fit(white_train_data_limit, white_train_labels)

    white_rf_train_labels = rf.predict(white_train_data_limit)
    white_rf_test_labels = rf.predict(white_test_data_limit)

    print(rf.feature_importances_)


    #RF model with alternate labels
    rf.fit(train_data_limit, train_alt_labels)
    
    alt_rf_train_labels = rf.predict(train_data_limit)
    alt_rf_test_labels = rf.predict(test_data_limit)
    
    print(rf.feature_importances_)


    return 0



#load raw data from CSV file
raw = pd.read_csv('recid_data.csv')

#handle non_numeric column types
df, race_codes = convert_non_numeric_data(raw)
print(df.head())

#output results from an initial analysis of the dataset
#init_analysis(df)

#model training
model_training(df, race_codes)



