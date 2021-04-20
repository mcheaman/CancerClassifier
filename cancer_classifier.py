# Name: Mason Heaman
# Date: April 20th, 2021
# Description: Train a Random Tree Classifier with tumor data determine whether a given test is malignant or benign
###############################################################################
import numpy as np
import csv
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import sklearn.preprocessing as pre
import warnings

def make_training_set(filename):
    """ Handle training data from the csv using pandas and numpy. Return numpy arrays for features and labels of the dataset
    Filter labels to 0-1 values to aid in classification.
    Precondition: filename is csv with a str label at [-1]
    """
    # Read in file
    # Separate csv into feature and label sets 
    data_set = pd.read_csv(filename)
    features = (data_set.iloc[:, :-1].astype(float)).to_numpy()
    labels = ((data_set.iloc[:, -1:].astype(str)).to_numpy()).reshape(-1,1)
    #encode labels to 0-1 values for classifier training
    encode = pre.LabelEncoder()
    labels = encode.fit_transform(labels)

    return features, labels

def precision(true, predicted):
    """Return the precision of the given predicted labels
    Args:
        true ([list]): The true labels provided in the dataset
        predicted ([list]): The labels predicted by classifier
    """
    # Create variables to track true positive and false positive
    tp = 0
    fp = 0
    for p,t in zip(true,predicted):
        if p == t and p == 1:
            tp += 1
        elif p != t and p == 1:
            fp += 1
    #Calculate precision
    return tp/(tp+fp)

def recall(true, predicted):
    """Return the recall of the given predicted labels
    Args:
        true ([list]): The true labels provided in the dataset
        predicted ([list]): The labels predicted by classifier
    """
    # Create variables to track true positive and false negative
    tp = 0
    fn = 0
    for p,t in zip(true,predicted):
        if p == t and p == 1:
            tp += 1
        elif p != t and p == 0:
            fn += 1
    # Calculate recall
    return tp/(tp+fn)

def fscore(true, predicted):
    """Return the fscore of the given predicted labels
    Args:
        true ([list]): The true labels provided in the dataset
        predicted ([list]): The labels predicted by classifier
    """
    p = precision(true,predicted)
    r = recall(true,predicted)
    # Calculate f-score
    return (2 * p * r) / (p + r)

def accuracy(true, predicted):
    """Return the accuaracy of the given predicted labels
    Args:
        true ([list]): The true labels provided in the dataset
        predicted ([list]): The labels predicted by classifier
    """
    # Create variables to track true positive, true negative, false positive, and false negative
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for p,t in zip(true,predicted):
        if p == t and p == 1:
            tp += 1
        elif p == t and p == 0:
            tn += 1
        elif p != t and p == 1:
            fp += 1
        else:
            fn += 1
    # Calculate accuracy
    return (tp + tn) / (tp + fp + tn + fn)


def classify(training_records, test_records):
    # Separate feature and label lists
    train_features = training_records[0]
    test_features = test_records[0]
    train_labels = training_records[1]
    test_labels = test_records[1]
    
    # Initialize classifier
    clf = RandomForestClassifier(n_estimators=10,max_depth=5).fit(train_features, train_labels)
    train_labels = np.array(train_labels)
    clf.fit(train_features, train_labels)

    # Predict Lables
    train_prediction = list(clf.predict(train_features))
    test_prediction = list(clf.predict(test_features))

    # Calculate training stats
    rtrain = recall(train_labels, train_prediction)
    ptrain = precision(train_labels, train_prediction)
    ftrain = fscore(train_labels, train_prediction)
    atrain = accuracy(train_labels, train_prediction)

    # Calculate testing stats
    rtest = recall(test_labels, test_prediction)
    ptest = precision(test_labels, test_prediction)
    ftest = fscore(test_labels, test_prediction)
    atest = accuracy(test_labels, test_prediction)
    
    # Performace for training and test sets
    train_performance = (rtrain, ptrain, ftrain, atrain)
    test_performance = (rtest, ptest, ftest, atest)

    return train_performance, test_performance

if __name__ == "__main__": 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Handle training set
        training_data_file = "cancerTrainingData.csv"
        training_set = make_training_set(training_data_file)
        
        # Handle test set
        test_file = "cancerTestingData.csv"
        test_set = make_training_set(test_file)
        performance = classify(training_set,test_set)
        
        # Report Performance
        print(f"Performance on Training Set\n\t Precision: {performance[0][1]}\n\t Recall: {performance[0][0]}\n\t F-score: {performance[0][2]}\n\t Accuracy: {performance[0][3]}")
        print(f"Performance on Testing Set\n\t Precision: {performance[1][1]}\n\t Recall: {performance[1][0]}\n\t F-score: {performance[1][2]}\n\t Accuracy: {performance[1][3]}")

