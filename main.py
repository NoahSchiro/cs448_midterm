import algos.Logistic_Reg as reg
from algos.bayes import Bayes 
from algos.utils import get_data

if __name__=="__main__":

    # Get the data
    data = get_data()

    # Run the regression script
    #reg.main(data)
    #svm.main(data)

    bayes = Bayes()
    bayes.train(data)

    bayes.test()
