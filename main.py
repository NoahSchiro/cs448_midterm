from algos.bayes import Bayes

word_mapping = {}
pos_mapping  = {}

def get_data():
    with open("train.txt", "r") as f:

        # Long string
        txt = f.read()

        # List of strings
        lines = txt.split('\n')

        complete_data = []

        for line in lines:
            # Split each line into a 3 tuple
            complete_data.append(tuple(line.split(' ')))

        return complete_data

def data_preprocess(data):

    # Filter out all training pieces that don't have 2 elements
    data = list(filter(lambda x: len(x) >= 2, data))

    # The data is composed as [(X_2, Y_2), (X_2, Y_2), ...]
    # So we will begin by splitting this into [[X_1], [X_2], ...]] and [Y_1, Y_2, ...]

    X = list(map(lambda x: x[0], data))
    Y = list(map(lambda x: x[1], data))
    print(X)

    return X, Y

def data_vectorize(data):

    word_counter = 0

    for (X, Y, _) in data:
        if X not in word_mapping:
            word_mapping[X] = word_counter
            word_counter += 1


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

