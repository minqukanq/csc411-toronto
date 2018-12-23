from hw1_code_question.check_grad import check_grad
from hw1_code_question.utils import *
from hw1_code_question.logistic import *
from hw1_code_question.plot_digits import *
import matplotlib.pyplot as plt

def run_logistic_regression(hyperparameters):
    # TODO specify training data
    # train_inputs, train_targets = load_train()
    train_inputs, train_targets = load_train()
    valid_inputs, valid_targets = load_valid()

    # N is number of examples; M is the number of features per example.
    N, M = train_inputs.shape
    valid_N, valid_M = valid_inputs.shape

    ones  = np.ones(N)
    valid_ones = np.ones(valid_N)
    train_inputs = np.insert(train_inputs, 0, ones, axis=1)
    valid_inputs = np.insert(valid_inputs, 0, valid_ones, axis=1)

    # Logistic regression weights
    # TODO:Initialize to random weights here.
    weights = np.random.randn(M+1, 1)
    # weights = np.zeros((M+1, 1))

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)
    # Begin learning with gradient descent
    logging = np.zeros((hyperparameters['num_iterations'], 5))
    for t in range(hyperparameters['num_iterations']):

        # Find the negative log likelihood and its derivatives w.r.t. the weights.
        f, df, predictions = logistic(weights, train_inputs, train_targets, hyperparameters)

        # Evaluate the prediction.
        cross_entropy_train, frac_correct_train = evaluate(train_targets, predictions)

        if np.isnan(f) or np.isinf(f):
            raise ValueError("nan/inf error")

        # update parameters
        weights = weights - hyperparameters['learning_rate'] * df / N

        # Make a prediction on the valid_inputs.
        predictions_valid = logistic_predict(weights, valid_inputs)

        # Evaluate the prediction.
        cross_entropy_valid, frac_correct_valid = evaluate(valid_targets, predictions_valid)

        # print some stats
        print("ITERATION:{:4d}  TRAIN NLOGL:{:4.2f}  TRAIN CE:{:.6f} "
               "TRAIN FRAC:{:2.2f}  VALID CE:{:.6f}  VALID FRAC:{:2.2f}",
                   t+1, float(f) / N, cross_entropy_train, frac_correct_train*100,
                   cross_entropy_valid, frac_correct_valid*100)
        logging[t] = [f / N, cross_entropy_train, frac_correct_train*100, cross_entropy_valid, frac_correct_valid*100]


    #TEST DATA
    test_inpurts, test_targets = load_test()
    test_ones = np.ones(np.shape(test_inpurts)[0])
    test_inpurts = np.insert(test_inpurts, 0, test_ones, axis=1)

    # Make a prediction on the valid_inputs.
    predictions_valid = logistic_predict(weights, test_inpurts)

    # Evaluate the prediction.
    cross_entropy_valid, frac_correct_valid = evaluate(test_targets, predictions_valid)

    # print some stats
    print("TEST CE:{:.6f}  TEST FRAC:{:2.2f}",cross_entropy_valid, frac_correct_valid*100 )

    return logging

def run_check_grad(hyperparameters):
    """Performs gradient check on logistic function.
    """

    # This creates small random data with 7 examples and 
    # 9 dimensions and checks the gradient on that data.
    num_examples = 7
    num_dimensions = 9

    weights = np.random.randn(num_dimensions+1, 1)
    data    = np.random.randn(num_examples, num_dimensions)
    targets = (np.random.rand(num_examples, 1) > 0.5).astype(int)

    N, M = np.shape(data)
    one  = np.ones(N)
    data = np.insert(data, 0, one, axis=1)

    diff = check_grad(logistic,      # function to check
                      weights,
                      0.001,         # perturbation
                      data,
                      targets,
                      hyperparameters)

    print("diff = %f ", diff)

if __name__ == '__main__':
    # TODO: Set hyperparameters
    hyperparameters = {
                    'learning_rate': 0.9,
                    'weight_regularization': 0, # boolean, True for using Gaussian prior on weights
                    'num_iterations': 1000,
                    'weight_decay': 1 # related to standard deviation of weight prior
                    }

    # average over multiple runs
    num_runs = 1
    logging = np.zeros((hyperparameters['num_iterations'], 5))
    for i in range(num_runs):
        logging += run_logistic_regression(hyperparameters)
    logging /= num_runs

    # TODO generate plots
    training_plot = plt.plot(logging[:, 1], 'r', label='training set')
    validation_plot = plt.plot(logging[:, 3], 'y', label='validation set')
    # plt.title('mnist_train_small')
    plt.title('mnist_train')
    plt.legend(loc='upper right')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Cross Entropy')
    plt.show()