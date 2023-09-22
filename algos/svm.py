from .utils import get_data, extract_features, prepare_data
from sklearn import svm
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report
def train_svm_model(t_features, t_labels):

    # Vectorize the features (give them numerical value)
    vectorizer = DictVectorizer()
    v_features = vectorizer.fit_transform(t_features)

    # Train SVM model
    clf = svm.SVC(kernel='linear')
    clf.fit(v_features, t_labels)

    return v_features, t_labels
    
def evaluate_model(model, vectorizer, data):
    dev_features, dev_labels = prepare_data(data)
    X_dev = vectorizer.transform(dev_features)
    predicted_labels = model.predict(X_dev)
    
    print(classification_report(dev_labels, predicted_labels))


def main(data):

    train_data, dev_data, test_data = get_data()

    train_features, train_labels = prepare_data(train_data)
    dev_features, dev_labels = prepare_data(dev_data)
    test_features, test_labels = prepare_data(test_data)

    #Train with SVM model

    svmodel, vectorizer = train_svm_model(train_features, train_labels)

    #Evaluate the model dev set
    print("Evaluation on the Development Set:")
    evaluate_model(svmodel, vectorizer, dev_data)

    # Evaluate the model on the test set
    print("Evaluation on the Test Set:")
    evaluate_model(svmodel, vectorizer, test_data)

if __name__== "__main__":
    data = get_data()
    main(data)

