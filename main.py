import algos.log_reg as log_reg
import algos.svm as svm
import algos.utils as utils
from algos.bayes import Bayes
from algos.utils import get_data

if __name__ == "__main__":
    # Get the data and split
    train_data, test_data = get_data(train_file="train.txt", test_file="unlabeled_test_test.txt")

    # Prepare the data for training and testing
    train_features, train_labels = utils.prepare_data(train_data)
    #test_features, test_labels = utils.prepare_data(test_data)

    # Train the Logistic Regression model
    print("-----------------TRAINING LOG REG------------------------")
    log_reg_model, log_reg_vectorizer = log_reg.train_logistic_regression_model(train_features, train_labels)

    # Train the SVM model
    print("-----------------TRAINING SVM------------------------")
    svm_model, svm_vectorizer = svm.train_svm_model(train_features, train_labels)

    # Run the bayes script
    print("-----------------TRAINING BAYES------------------------")
    bayes = Bayes()
    bayes.train(train_data)

    total = 0
    correct = 0

    # Open the output file for writing
    with open("NLP_Pioneers.test.txt", "w") as output_file:
        for feature in test_data:

            # Make a prediction for bayes
            b_pred = bayes.forward(feature)

            # Make a prediction for svm
            s_pred = svm.predict_pos_tags([feature["token"]], svm_vectorizer, svm_model)[0]

            # Make a prediction for Logistic Regression
            l_pred = log_reg.predict_pos_tags([feature["token"]], log_reg_vectorizer, log_reg_model)[0]

            pred = None

            # If they all predict the same thing, use that
            if b_pred == s_pred and s_pred == l_pred:
                pred = b_pred

            # The following rules just implement 2/3rds majority guess
            elif b_pred == s_pred or b_pred == l_pred:
                pred = b_pred
            elif s_pred == l_pred:
                pred = s_pred

            # If NONE of them equal each other, then just use the model
            # with the highest accuracy
            else:
                pred = l_pred

            # Write the token and predicted tag to the output file
            output_file.write(f"{feature['token']},{pred}\n")

        
