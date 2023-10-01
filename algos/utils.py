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

def get_data(test_size=0.20, random_seed=42):
    with open("train.txt", "r") as f:
        lines = f.read().strip().split('\n')
        data = [line.split(' ')[:2] for line in lines]

    # Split the data into training, dev, and test sets
    random.seed(random_seed)
    random.shuffle(data)

    test_split = int(len(data) * test_size)

    test_data = data[:test_split]
    train_data = data[test_split:]

    return train_data, test_data

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
    data = get_data()

    # Print the first few tuples of the data
    print("Sample data tuples:")
    for i in range(min(len(data), 5)):  # Print the first 5 tuples
        print(data[i])


    # Prepare the data
    features, labels = prepare_data(data)

