## tip for code to work: run in term "pip3 install scikit-learn" and " pip3 install nltk"
#from .utils import get_data, extract_features, prepare_data

# Import for vectorizing features
from sklearn.feature_extraction import DictVectorizer
# Import for logistic regression model
from sklearn.linear_model import LogisticRegression
# Import for generating classification report
from sklearn.metrics import classification_report
#---------
# tip for code to work: run in term "pip3 install scikit-learn"
import random
# Import for generating classification report
from sklearn.metrics import classification_report

# Import for data splitting
from sklearn.model_selection import train_test_split  

# We only need the first two columns (token and POS tag),
# modifying the code slightly to return only those columns.
# get_data() reads and parses the training data from 'train.txt' and returns a list of tuples
# containing token and POS tag.

def get_data(test_size=0.15, dev_size=0.15, random_seed=42):
    with open("train.txt", "r") as f:
        lines = f.read().strip().split('\n')
        data = [line.split(' ')[:2] for line in lines]

    # Split the data into training, dev, and test sets
    random.seed(random_seed)
    random.shuffle(data)

    test_split = int(len(data) * test_size)
    dev_split = int(len(data) * dev_size)

    test_data = data[:test_split]
    dev_data = data[test_split:test_split + dev_split]
    train_data = data[test_split + dev_split:]

    return train_data, dev_data, test_data

# Feature extraction: defining features for each token based on its context,
# such as the previous and next words, prefixes, suffixes, etc. 
# extract_features() extracts features for a given token in a sentence. In this example, it uses the
# current token as a feature.

def extract_features(token):
    #token = sentence[index][0]
    return {'token': token}

# Prepare Training Data:  creating feature vectors and corresponding
# labels (POS tags) for logistic regression model using 
# import nltk
# prepare_data() Prepares the training data by extracting features and labels (POS tags) for
# the Logistic Regression model.

def prepare_data(data):
    features = []
    labels = []

    for sentence in data:
        token = sentence[0]  # Assuming the token is the first (and only) element
        pos_tag = sentence[1] if len(sentence) > 1 else ''  # Use the second element if available, or an empty string if not
        features.append(extract_features(token))
        labels.append(pos_tag)

    return features, labels


#-------
# need to install scikit-learn (type "pip install scikit-learn"
# in terminal)

# Training Logistic Regression Model: using se the LogisticRegression
# class from the sklearn library to train model. 
# install scikit-learn 
# train_logistic_regression_model() Trains a Logistic Regression model
# using the provided features and labels.

def train_logistic_regression_model(features, labels):
    # Vectorize the features
    vectorizer = DictVectorizer()
    X = vectorizer.fit_transform(features)

    # Train a Logistic Regression model
    clf = LogisticRegression(max_iter=1000, verbose=1)
    clf.fit(X, labels)
    
    return clf, vectorizer

# Evaluation using cross validation or not...
# evaluate_model() evaluates the trained model using a small part 
# of the training data as a dev set and prints the classification report.

def evaluate_model(model, vectorizer, data):
    dev_features, dev_labels = prepare_data(data)
    X_dev = vectorizer.transform(dev_features)
    predicted_labels = model.predict(X_dev)
    
    print(classification_report(dev_labels, predicted_labels))

# Predicting POS tags: pred POS tags for new sentences
# using the trained calssifier

def predict_pos_tags(sentence, vectorizer, clf):
    features = [extract_features(token) for token in sentence]
    X = vectorizer.transform(features)
    predicted_labels = clf.predict(X)
    return predicted_labels


def main(data):
     # Load and preprocess the data
    train_data, dev_data, test_data = get_data()

    # Prepare the data for training, dev, and testing
    train_features, train_labels = prepare_data(train_data)
    dev_features, dev_labels = prepare_data(dev_data)
    test_features, test_labels = prepare_data(test_data)

    # Train the Logistic Regression model
    model, vectorizer = train_logistic_regression_model(train_features, train_labels)

    # Evaluate the model on the development set
    print("Evaluation on the Development Set:")
    evaluate_model(model, vectorizer, dev_data)

    # Evaluate the model on the test set
    print("Evaluation on the Test Set:")
    evaluate_model(model, vectorizer, test_data)

if __name__ == "__main__":
    data = get_data()
    main(data)
