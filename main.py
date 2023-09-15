# This function read and parse the training data from "train.txt."
# It returns a list of tuples where each tuple contains the token, 
# POS tag, and Chunking tag. 
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

if __name__=="__main__":
    data = get_data()


