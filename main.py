import algos.Logistic_Reg as reg
from algos.utils import get_data

if __name__=="__main__":

    # Get the data
    data = get_data()

    # Run the regression script
    reg.main(data)

    #bayes.main(data)

    #svm.main(data)

