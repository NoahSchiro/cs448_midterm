from algos.bayes import Bayes
from algos.utils import *

if __name__=="__main__":

    # Retrieve data
    data = get_data()

    # Calculate mappings
    data_vectorize(data) 

    # Perform a train / validate split of 80/20
    idx = int(len(data) * 0.8)
    train_ds, val_ds = data[:idx], data[idx:]

    X_train, Y_train = data_preprocess(train_ds)
    X_val, Y_val = data_preprocess(val_ds)

    # Intialize models
    b = Bayes()
    # s = SVM()
    # r = Regression()

    b.train(data)
