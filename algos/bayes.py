from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction import DictVectorizer

from .utils import prepare_data

class Bayes():

    # Intialize model
    def __init__(self):
        
        # Intialize vectorizer and naive bayes algo
        self.vec = DictVectorizer()
        self.model = GaussianNB()

    def pre_process(self, data):

        # Train, dev, test split
        train, dev, test = data

        # Split into features and labels
        self.X_train, self.Y_train = prepare_data(train)
        self.X_dev,   self.Y_dev   = prepare_data(dev)
        self.X_test,  self.Y_test  = prepare_data(test)

    # Training the model
    def train(self, data):

        # Run preprocessing on the data
        self.pre_process(data)

        # word -> vector
        self.X_train_vec = self.vec.fit_transform(self.X_train)

        # Get a list of all possible classes (needed for training)
        possible_classes = list(set(self.Y_train))

        # We need to process this in batches because there is so much data
        for idx in range(0, len(self.Y_train), 5000):

            progress = (idx / len(self.Y_train)) * 100
            print(f"Training... {progress:.2f}%", end="\r")

            # Get batch
            X = self.X_train_vec[idx:idx+5000]
            Y = self.Y_train[idx:idx+5000]

            # Must tell model possible classes on first iteration
            # May be ommitted in future iterations
            if idx == 0:
                self.model.partial_fit(X.toarray(), Y, possible_classes)
            else:
                self.model.partial_fit(X.toarray(), Y)

        print("Training complete!")


    def test(self):

        correct = 0
        total = 0

        X = self.vec.transform(self.X_test)
        Y = self.Y_test

        for idx in range(0, len(Y), 5000):
            
            X_batch = X[idx:idx+5000]
            Y_batch = Y[idx:idx+5000]

            # making predictions on the testing set
            Y_pred = self.model.predict(X_batch.toarray())

            for (p, y) in zip(Y_pred, Y_batch):
                total += 1
                if p == y:
                    correct += 1

            progress = (idx / len(self.Y_test)) * 100
            print(f"Testing... {progress:.2f}%; acc = {(correct/total) * 100:.2f}", end="\r")

        print("\n ")
        print(f"Accuracy: {(correct / total) * 100:.2f}%")


          
    # Forward is like our inference
    def forward(self, x):
        return self.model.predict(x)
