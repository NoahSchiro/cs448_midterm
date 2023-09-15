from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

class Bayes():

    # Intialize model
    def __init__(self):
        self.model = GaussianNB()

    # Training the model
    def train(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def test(self, X_test, Y_test):

        # making predictions on the testing set
        Y_pred = self.model.predict(X_test)
          
        # comparing actual response values (y_test) with predicted response values (y_pred)
        print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(Y_test, Y_pred)*100)

    # Forward is like our inference
    def forward(self, x):
        return self.model.predict(x)
