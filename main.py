import algos.log_reg as log_reg
import algos.svm as svm
from algos.bayes import Bayes 
from algos.utils import get_data

if __name__=="__main__":

    # Get the data
    data = get_data()

    # Run the regression script
    #log_reg.main(data)

    # Run the svm script
    svm.main(data)

    # Run the bayes script
    #bayes = Bayes()
    #bayes.train(data)

    #bayes.test()
