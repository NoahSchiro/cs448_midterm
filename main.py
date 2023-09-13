
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

