# cs448_midterm
---

Split the three algorithms among the three of us:

- [x] Bayesian Classifier
- [x] SVM
- [x] Linear / Logisitc Regression (for this task, likely a logisitic regression)

Ideas for maximizing performance:
I don't think it'll be enough to just pass in the word and expect it to predict the part of speech, we might want to incorporate positional encoding as well (where the word is in the sentence) or maybe what words came before or after the word?

## Linear/Logistic Regression
---

Here is the explanation of how the logistic regression for POS tagging. Here's a brief overview of its implementation:

Data Preparation: The code reads training data from "train.txt," which contains sentences with tokens and their corresponding POS tags. It extracts the token and POS tag from each line and stores them in a list of tuples, where each tuple contains a token and its POS tag.

Feature Extraction: It defines a feature extraction function extract_features(token) that extracts features for each token. In the provided code, it uses the token itself as a feature.

Data Vectorization: It vectorizes the features using scikit-learn's DictVectorizer. This step converts the feature dictionaries into a numerical format that can be used for training a machine learning model.

Logistic Regression Model Training: It trains a logistic regression model using scikit-learn's LogisticRegression class. The features and corresponding labels (POS tags) are used for training the model.

Model Evaluation: It evaluates the trained model using a small part of the training data as a dev set. It calculates and prints the classification report, which includes metrics such as precision, recall, and F1-score for each POS tag.

Prediction: Finally, it provides a function predict_pos_tags(sentence, vectorizer, clf) that allows you to predict POS tags for new sentences using the trained classifier.
