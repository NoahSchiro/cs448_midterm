from utils import *

# Import for logistic regression model
from sklearn.linear_model import LogisticRegression
# Import for vectorizing features
from sklearn.feature_extraction import DictVectorizer

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

# Predicting POS tags: pred POS tags for new sentences
# using the trained calssifier
def predict_pos_tags(sentence, vectorizer, clf):
    features = [extract_features(sentence, i) for i in range(len(sentence))]
    X = vectorizer.transform(features)
    predicted_labels = clf.predict(X)
    return predicted_labels


def main():
    # Load and preprocess the data
    data = get_data()

    # Prepare the data
    features, labels = prepare_data(data)

    # Train the Logistic Regression model
    model, vectorizer = train_logistic_regression_model(features, labels)

    # Evaluate the model
    evaluate_model(model, vectorizer, data)

if __name__ == "__main__":
    main()
