## tip for code to work: run in term "pip3 install scikit-learn" and " pip3 install nltk"

import nltk # natural language toolkit
#nltk.download('punkt')

# Import for vectorizing features
from sklearn.feature_extraction import DictVectorizer
# Import for logistic regression model
from sklearn.linear_model import LogisticRegression
# Import for generating classification report
from sklearn.metrics import classification_report

# need to install scikit-learn (type "pip install scikit-learn"
# in terminal)



# We only need the first two columns (token and POS tag),
# modifying the code slightly to return only those columns.
# get_data() reads and parses the training data from 'train.txt' and returns a list of tuples
# containing token and POS tag.

def get_data():
    with open("train.txt", "r") as f:
        lines = f.read().strip().split('\n')
        data = [line.split(' ')[:2] for line in lines]
    return data

# Featue extraction: defining features for each token based on its context,
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
    clf = LogisticRegression(max_iter=1000)
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


def main():
    # Load and preprocess the data
    data = get_data()

    # Print the first few tuples of the data
    print("Sample data tuples:")
    for i in range(min(len(data), 5)):  # Print the first 5 tuples
        print(data[i])


    # Prepare the data
    features, labels = prepare_data(data)

    # Train the Logistic Regression model
    model, vectorizer = train_logistic_regression_model(features, labels)

    # Evaluate the model
    evaluate_model(model, vectorizer, data)

if __name__ == "__main__":
    main()