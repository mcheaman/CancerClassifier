# Name: Mason Heaman
# Date: November 26th, 2019
# Description: A program that learns from tumor data to determine whether a given test is malignant or benign
###############################################################################
# For use as dictionary keys
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
    # COMPLETE - DO NOT MODIFY
    training_records = []
    # Read in file
    for line in open(filename,'r'):
        if '#' in line:
            continue
        line = line.strip('\n')
        line_list = line.split(',')
        
        # Create a dictionary for the line and map the attributes in
        # ATTRS to the corresponding values in the line of the file
        record = {}
        
        # read patient ID as an int:
        record[ATTRS[0]] = int(line_list[0].strip())
        
        # read attributes 1 through 10 as floats:
        for i in range(1,11):
            record[ATTRS[i]] = float(line_list[i])
        
        # read the class (label), which is "M", or "B" as a string:
        record[ATTRS[11]] = line_list[31].strip() 

        # Add the dictionary to a list
        training_records.append(record)        

    return training_records


def make_test_set(filename):
    """ Read test data from the file whose path is filename.
        Return a list with the same form as the training
        set, except that each dictionary has an additional
        key "prediction" initialized to "none" that will be
        used to store the label predicted by the classifier. 
    """
    # COMPLETE - DO NOT MODIFY
    test_records = make_training_set(filename)

    for record in test_records:
        record["prediction"] = "none"

    return test_records

def midpoint(a, b):
    """ Return the midpoint between two values """
    midpoint = (a + b) /2
    return midpoint

def average_attr(attr_list):
    """ Return an average for each attribute using its list of data.
    Precondition: arguments are lists.
    Postcondition: a single float will be returned for the average
    """
    total = 0
    for i in range(0,len(attr_list)):
        total += attr_list[i]
        i += 1
    average = total / len(attr_list)
    return average

def train_classifier(training_records):
    """ Return a dict containing the midpoint between averages
        among each class (malignant and benign) of each attribute.
        (See the A5 writeup for a more complete description)
        Precondition: training_records is a list of patient record
                      dictionaries, each of which has the keys
                      in the global variable ATTRS
        Postcondition: the returned dict has midpoint values calculated
                       from the training set for all 10 attributes except
                       "ID" and"class".
    """
    # TODO 1 - implement this function
    # create dictionaries for classifier, malignant, and benign
    classifier = {}
    malignant_list = []
    benign_list = []
    # create lists for attributes depending on malignant or benign
    radius_m = []
    texture_m = []
    perimeter_m = []
    area_m = []
    smoothness_m = []
    compactness_m = []
    concavity_m = []
    concave_m = []
    symmetry_m = []
    fractal_m = []
    radius_b = []
    texture_b = []
    perimeter_b = []
    area_b = []
    smoothness_b = []
    compactness_b = []
    concavity_b = []
    concave_b = []
    symmetry_b = []
    fractal_b = []

    #seperate into malignant and benign dicts
    for i in range(len(training_records)):
        if training_records[i]["class"]== "M":
                malignant_list.append(training_records[i])
        else:
                benign_list.append(training_records[i])
    for i in range(len(malignant_list)):
        radius_m.append(malignant_list[i][ATTRS[1]])
        texture_m.append(malignant_list[i][ATTRS[2]])
        perimeter_m.append(malignant_list[i][ATTRS[3]])
        area_m.append(malignant_list[i][ATTRS[4]])
        smoothness_m.append(malignant_list[i][ATTRS[5]])
        compactness_m.append(malignant_list[i][ATTRS[6]])
        concavity_m.append(malignant_list[i][ATTRS[7]])
        concave_m.append(malignant_list[i][ATTRS[8]])
        symmetry_m.append(malignant_list[i][ATTRS[9]])
        fractal_m.append(malignant_list[i][ATTRS[10]])
    for i in range(len(benign_list)):
        radius_b.append(benign_list[i][ATTRS[1]])
        texture_b.append(benign_list[i][ATTRS[2]])
        perimeter_b.append(benign_list[i][ATTRS[3]])
        area_b.append(benign_list[i][ATTRS[4]])
        smoothness_b.append(benign_list[i][ATTRS[5]])
        compactness_b.append(benign_list[i][ATTRS[6]])
        concavity_b.append(benign_list[i][ATTRS[7]])
        concave_b.append(benign_list[i][ATTRS[8]])
        symmetry_b.append(benign_list[i][ATTRS[9]])
        fractal_b.append(benign_list[i][ATTRS[10]])
    #find midpoints between averages
    classifier["radius"]= midpoint((average_attr(radius_m)), (average_attr(radius_b)))
    classifier["texture"]= midpoint((average_attr(texture_m)), (average_attr(texture_b)))
    classifier["perimeter"]= midpoint((average_attr(perimeter_m)), (average_attr(perimeter_b)))
    classifier["area"]= midpoint((average_attr(area_m)), (average_attr(area_b)))
    classifier["smoothness"]= midpoint((average_attr(smoothness_m)), (average_attr(smoothness_b)))
    classifier["compactness"]= midpoint((average_attr(compactness_m)), (average_attr(compactness_b)))
    classifier["concavity"]= midpoint((average_attr(concavity_m)), (average_attr(concavity_b)))
    classifier["concave"]= midpoint((average_attr(concave_m)), (average_attr(concave_b)))
    classifier["symmetry"]= midpoint((average_attr(symmetry_m)), (average_attr(symmetry_b)))
    classifier["fractal"]= midpoint((average_attr(fractal_m)),(average_attr(fractal_b)))
    
    return classifier

def classify(test_records, classifier):
    """ Use the given classifier to make a prediction for each record in
        test_records, a list of dictionary patient records with the keys in
        the global variable ATTRS. A record is classified as malignant
        if at least 5 of the attribute values are above the classifier's
        threshold.
        Precondition: classifier is a dict with midpoint values for all
                      keys in ATTRS except "ID" and "class"
        Postcondition: each record in test_records has the "prediction" key
                       filled in with the predicted class, either "M" or "B"
    """
    # TODO 2 - implement this function

    for i in range(len(test_records)):
        malignant_vote = 0
        benign_vote = 0
        for j in range(1,11):
            if test_records[i][ATTRS[j]] <= classifier[ATTRS[j]]:
                benign_vote += 1
            else:
                malignant_vote += 1
        if malignant_vote >= benign_vote:
            test_records[i]["prediction"] = "M"
        else:
            test_records[i]["prediction"] = "B"
        
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
    ID = int(input("Enter a patient ID to see classification details:"))
    found = 0
    # while the user has not entered "quit":
    while ID != "quit":
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
     training_data_file = "cancerTrainingData.txt"
     training_set = make_training_set(training_data_file)
     print("Done reading training data.")
    
    # load the test set 
     print("Reading in test data...")
     test_file = "cancerTestingData.txt"
     test_set = make_test_set(test_file)
     print("Done reading test data.\n")

    # train the classifier: uncomment this block once you've
    # implemented train_classifier
     print("Training classifier..."    )
     classifier = train_classifier(training_set)
     print("Classifier cutoffs:")
     for key in ATTRS[1:11]:
         print("    ", key, ": ", classifier[key], sep="")
     print("Done training classifier.\n")

    # use the classifier to make predictions on the test set:
    # uncomment the following block once you've written classify
    # and report_accuracy
     print("Making predictions and reporting accuracy")
     classify(test_set, classifier)
     report_accuracy(test_set)
     print("Done classifying.\n")

    # prompt the user for patient IDs and provide details on
    # the diagnosis: uncomment this line when you've
    # implemented check_patients
     check_patients(test_set, classifier)
