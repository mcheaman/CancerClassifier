# Name: Mason Heaman
# Date: April 12th, 2021
# Description: A program that learns from tumor data to determine whether a given test is malignant or benign
###############################################################################
# For use as dictionary keys
import numpy as np
import csv
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import sklearn.preprocessing as pre
ATTRS = []
ATTRS.append("ID")
ATTRS.append("radius")
ATTRS.append("texture")
ATTRS.append("perimeter")
ATTRS.append("area")
ATTRS.append("smoothness")
ATTRS.append("compactness")
ATTRS.append("concavity")
ATTRS.append("concave")
ATTRS.append("symmetry")
ATTRS.append("fractal")
ATTRS.append("class")
###############################################################################


def make_training_set(filename):
    """ Read training data from the file whose path is filename.
        Return a list of records, where each record is a dictionary
        containing a value for each of the 12 keys in ATTRS.
    """
    # Read in file

    data_set = pd.read_csv('cancerTrainingData.csv')
    features = data_set.iloc[:, :-1].astype(int)
    labels = data_set.iloc[:, -1:].astype(str)

    return features, labels



def classify(training_records, test_records):
    train_features = training_records[0].to_numpy()
    train_labels = training_records[1].to_numpy()
    test_features = test_records[0].to_numpy()
    test_labels = test_records[1].to_numpy()

    # print(train_labels)
    train_labels = train_labels.ravel()
    # train_labels = train_labels.reshape(-1,1)
    test_labels =test_labels.ravel()
    # test_labels = train_labels.reshape(-1,1)
    
    encode = pre.LabelEncoder()
    train_labels = encode.fit_transform(train_labels)
    test_labels = encode.fit_transform(test_labels)
    train_features = train_features.tolist() 
    train_features = test_features.tolist() 
    train_features = encode.fit_transform(train_features)
    test_features = encode.fit_transform(test_features)
    # train_labels = train_labels.reshape(-1,1)
    # test_labels = train_labels.reshape(-1,1)

    clf = RandomForestClassifier(n_estimators=10,max_depth=5).fit(train_features, train_labels)

    train_prediction = clf.predict(train_features)
    test_prediction = clf.predict(test_features)
    print(train_labels)

        
def id_classify(test_records, classifier, ID, i, index):
    """A more specific classifier that gives the vote on a single attribute
    Postcondition: the returned dictionary will contain votes for the attributes in "Malignant" or "Benign"
    """
    if (test_records[index][ATTRS[i]]) <= (classifier[ATTRS[i]]):
        id_vote = "Benign"
    else:
        id_vote = "Malignant"
    
    return id_vote

def report_accuracy(test_records):
    """ Print the accuracy of the predictions made by the classifier
        on the test set as a percentage of correct predictions.
        Precondition: each record in the test set has a "prediction"
        key that maps to the predicted class label ("M" or "B"), as well
        as a "class" key that maps to the true class label. """
    # TODO 3 - implement this function
    accurate = 0
    total = len(test_records)
    for i in range(len(test_records)):
        if (test_records[i]["prediction"]) == (test_records[i]["class"]):
            accurate += 1
    accuracy = (accurate / total)
    print(accuracy)
        
def chart(test_records, classifier, ID,index):
    """ Creates a chart for the results of check_patients.
    Precondition: The patient ID is present in test_records
    Postcondition: The charts rows are inline nad right justified
    """
    

    print(test_records[index][ATTRS[0]])
    print("Attribute".rjust(15), "Patient".rjust(10), "Classifier".rjust(10), "Vote".rjust(10), end="\n")
    for i in range (1,11):
        print((ATTRS[i]).rjust(15), "{:.4f}".format(test_records[index][ATTRS[i]]).rjust(10), "{:.4f}".format(classifier[ATTRS[i]]).rjust(10), id_classify(test_records, classifier, ID, i, index).rjust(10))
    print("Classifier's Diagnosis: ", test_records[index]["prediction"])


def check_patients(test_records, classifier):
    """ Repeatedly prompt the user for a Patient ID until the user
        enters "quit". For each patient ID entered, search the test
        set for the record with that ID, print a message and prompt
        the user again. If the patient is in the test set, print a
        table: for each attribute, list the name, the patient's value,
        the classifier's midpoint value, and the vote cast by the
        classifier. After the table, output the final prediction made
        by the classifier.
        If the patient ID is not in the test set, print a message and
        repeat the prompt. Assume the user enters an integer or quit
        when prompted for the patient ID.
    """
    
    # prompt user for an ID
    ID = int(input("Enter a patient ID to see classification details: "))
    found = 0
    # while the user has not entered "quit":
    while ID != "quit" or ID != "q":
        for i in range(len(test_records)):
            if ID in test_records[i].values():
                 found = 1
        
                
        if found == 1:
            for j in range(len(test_records)):
                if ID ==(test_records[j][ATTRS[0]]):
                    index = j
            chart(test_records, classifier, ID, index)
        else:
            print("The ID was not found")
        ID = input("Enter a patient ID to see classification details:") 


if __name__ == "__main__": 
    # Main program - COMPLETE
    # Do not modify except to uncomment each code block as described.
    
    # load the training set
    print("Reading in training data...")
    training_data_file = "cancerTrainingData.csv"
    training_set = make_training_set(training_data_file)
    print("Done reading training data.")
    
    # load the test set 
    print("Reading in test data...")
    test_file = "cancerTestingData.csv"
    test_set = make_training_set(test_file)
    print("Done reading test data.\n")

    classify(training_set,test_set)

    # # train the classifier: uncomment this block once you've
    # # implemented train_classifier
    #  print("Training classifier..."    )
    #  classifier = train_classifier(training_set)
    #  print("Classifier cutoffs:")
    #  for key in ATTRS[1:11]:
    #      print("    ", key, ": ", classifier[key], sep="")
    #  print("Done training classifier.\n")

    # # use the classifier to make predictions on the test set:
    # # uncomment the following block once you've written classify
    # # and report_accuracy
    #  print("Making predictions and reporting accuracy")
    #  classify(test_set, classifier)
    #  report_accuracy(test_set)
    #  print("Done classifying.\n")

    # # prompt the user for patient IDs and provide details on
    # # the diagnosis: uncomment this line when you've
    # # implemented check_patients
    #  check_patients(test_set, classifier)
