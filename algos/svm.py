from .utils import get_data, extract_features, prepare_data

#Import SVM Model
from sklearn import svm
#Import Vectorizer to Vectorize Features
from sklearn.feature_extraction import DictVectorizer
#Generate Classification Report
from sklearn.metrics import classification_report

# Train an SVM model using the provided training data.
def train_svm_model(t_features, t_labels):
    # Vectorize the features (give them numerical value)
    vectorizer = DictVectorizer()
    v_features = vectorizer.fit_transform(t_features)
    
    # Train SVM model
    clf = svm.SVC(kernel='linear', verbose=True)
    clf.fit(v_features,t_labels)
    
    return clf, vectorizer
    
    
# Evaluate the SVM model on a dataset.
def evaluate_model(model, vectorizer, data):
    dev_features, dev_labels = prepare_data(data)
    X_dev = vectorizer.transform(dev_features)
    predicted_labels = model.predict(X_dev)
    
    print("[SVM] Classification Report:\n", classification_report(dev_labels, predicted_labels))

# Predict part-of-speech tags for a sentence using the trained SVM model.
def predict_pos_tags(sentence, vectorizer, clf):
    features = [extract_features(token) for token in sentence]
    X = vectorizer.transform(features)
    predicted_labels = clf.predict(X)
    return predicted_labels

def main(data):

    print("[SVM] Loading data...")
    train_data, test_data = get_data()
    
    # Prepare training, development, and test data
    train_features, train_labels = prepare_data(train_data)

    print("[SVM] Data load done...")

    # Train with SVM model
    svmodel, vectorizer = train_svm_model(train_features, train_labels)

    # Evaluate the model on the test set
    print("[SVM] Evaluation on the Test Set:")
    evaluate_model(svmodel, vectorizer, test_data)
    
    # Example sentence for prediction
    sentence_to_predict = ["This", "is", "an", "example", "sentence", "."]

    # Predict part-of-speech tags for the example sentence
    predicted_tags = predict_pos_tags(sentence_to_predict, vectorizer, svmodel)
    print("[SVM] Predicted Tags for the Example Sentence:", predicted_tags)

if __name__== "__main__":
    data = get_data()
    main(data)

