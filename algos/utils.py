# tip for code to work: run in term "pip3 install scikit-learn"
import random
# Import for generating classification report
from sklearn.metrics import classification_report
# Import for vectorizing features
from sklearn.feature_extraction import DictVectorizer

# Import for data splitting
from sklearn.model_selection import train_test_split  

# We only need the first two columns (token and POS tag),
# modifying the code slightly to return only those columns.
# get_data() reads and parses the training data from 'train.txt' and returns a list of tuples
# containing token and POS tag.

def get_data(train_file="train.txt", test_file="unlabeled_test_test.txt", test_size=0.10, random_seed=42):
    # Read and parse the training data from 'train.txt'
    with open(train_file, "r") as f:
        train_lines = f.read().strip().split('\n')
        train_data = [line.split(' ')[:2] for line in train_lines]

    # Read and parse the test data from 'unlabeled_test_test.txt'
    with open(test_file, "r") as f:
        test_lines = f.read().strip().split('\n')
        test_data = [line.split(' ')[:2] for line in test_lines]

    # Split the training data into training and validation sets
    random.seed(random_seed)
    random.shuffle(train_data)

    test_split = int(len(train_data) * test_size)

    validation_data = train_data[:test_split]
    train_data = train_data[test_split:]

    return train_data, validation_data, test_data

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

if __name__ == "__main__":

    # Load and preprocess the data
    train_data, validation_data, test_data = get_data()

    # Print the first few tuples of the training data
    print("Sample training data tuples:")
    for i in range(min(len(train_data), 5)):  # Print the first 5 tuples
        print(train_data[i])

    # Print the first few tuples of the test data
    print("Sample test data tuples:")
    for i in range(min(len(test_data), 5)):  # Print the first 5 tuples
        print(test_data[i])

    # Prepare the data
    train_features, train_labels = prepare_data(train_data)
    validation_features, validation_labels = prepare_data(validation_data)
    test_features, _ = prepare_data(test_data)

