# tip for code to work: run in term "pip3 install scikit-learn" and " pip3 install nltk"
# need to install scikit-learn (type "pip install scikit-learn" in terminal)

# Import for generating classification report
from sklearn.metrics import classification_report
# Import for vectorizing features
from sklearn.feature_extraction import DictVectorizer

# We only need the first two columns (token and POS tag),
# modifying the code slightly to return only those columns.
# get_data() reads and parses the training data from 'train.txt' and returns a list of tuples
# containing token and POS tag.
def get_data():
    with open("train.txt", "r") as f:
        lines = f.read().strip().split('\n')
        data = [line.split(' ')[:2] for line in lines]

    # Some of the data only contains ' ', so pad this with another so everything is length 2
    for line in data:
        if len(line) == 1:
            line.append(' ')
    return data

# Featue extraction: defining features for each token based on its context,
# such as the previous and next words, prefixes, suffixes, etc. 
# extract_features() extracts features for a given token in a sentence. In this example, it uses the
# current token as a feature.

def extract_features(sentence, index):
    token = sentence[index][0]
    return {'token': token}

# debugging: In this updated function,
# we assume that sentence is a list of tuples, 
# and we access the token using sentence[index][0]. 
# This should resolve the TypeError issue and yet........


# Prepare Training Data:  creating feature vectors and corresponding
# labels (POS tags) for logistic regression model using 
# import nltk
# prepare_data() Prepares the training data by extracting features and labels (POS tags) for
# the Logistic Regression model.
def prepare_data(data):
    features = []
    labels = []


    for sentence in data:
        for token, pos_tag, *_ in sentence:
            features.append(extract_features(sentence, token))  # Use token instead of data
            labels.append(pos_tag)

    return features, labels


# Evaluation using cross validation or not...
# evaluate_model() evaluates the trained model using a small part 
# of the training data as a dev set and prints the classification report.
def evaluate_model(model, vectorizer, data):
    dev_features, dev_labels = prepare_data(data)
    X_dev = vectorizer.transform(dev_features)
    predicted_labels = model.predict(X_dev)
    
    print(classification_report(dev_labels, predicted_labels))

def data_dictionary(data):
    X_data = list(map(lambda x: x[0], data))
    Y_data = list(map(lambda x: x[1], data))

    X_dict = {}
    Y_dict = {}
    counter = 0

    for elem in X_data:
        if elem not in X_dict:
            X_dict[elem] = counter
            counter += 1

    counter = 0

    for elem in Y_data:
        if elem not in Y_dict:
            Y_dict[elem] = counter
            counter += 1

    return X_dict, Y_dict

if __name__ == "__main__":

    # Get raw data
    data = get_data()

    # Get a mapping for 
    # word    -> integer and
    # POS tag -> integer
    xd, yd = data_dictionary(data)

    # Finally obtain a vectorization of our data
    x_vec = DictVectorizer().fit([xd])
    y_vec = DictVectorizer().fit([yd])
    

