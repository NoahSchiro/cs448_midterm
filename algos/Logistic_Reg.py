## tip for code to work: run in term "pip3 install scikit-learn" and " pip3 install nltk"
from utils import get_data, extract_features, prepare_data

# Import for vectorizing features
from sklearn.feature_extraction import DictVectorizer
# Import for logistic regression model
from sklearn.linear_model import LogisticRegression
# Import for generating classification report
from sklearn.metrics import classification_report

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
