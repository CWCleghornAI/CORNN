import lib.CORNN as CORNN
import numpy as np
if __name__ == "__main__":
  
    function_dictionary=CORNN.get_benchmark_functions()
    # Returns a dictionary of all the regression functions within CORNN
    # The key is the function name, the value is a 3 element tuple:
    # containing the raw objective function (for example the Ackley function) and
    # the x variable's domain and the y variable's domain

    print([*function_dictionary.keys()])
    # list all the available objective functions.

    training_data, test_data= CORNN.get_scaled_function_data(function_dictionary["Ackley"])
    # Both the training data and the test data are pairs. 
    # The first element of training_data is an PyTorch tensor of data patterns
    # The second element of training_data is a PyTorch tensor of the corresponding labels
    # The same is true for test_data

    neural_network_dictionary=CORNN.get_NN_models()
    # Returns a dictionary of all neural network architecture from CORNN
    # The key is the class name, the value is the NN class built on PyTorch. 

    print([*neural_network_dictionary.keys()])
    # list all the available neural network architecture.

    # This example is the same as in Demo.py but with an inline custom architecture.
    # This customization is not needed to use the proposed benchmark suit,
    # but has been included if you wish to explore ideas
    import torch
    import torch.nn as nn
    import torch.nn.functional as F 
    class Net_Custom(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(2, 10)     # input->H1
            self.fc2 = nn.Linear(10, 50)    # H1->H2
            self.fc3 = nn.Linear(50, 10)    # H2->H3
            self.fc4 = nn.Linear(10, 1)     # H3->output

        def forward(self, x):
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            x = torch.tanh(x)
            x = self.fc3(x)
            x = F.relu(x)
            x = self.fc4(x)
            return x
    neural_network_architecture=Net_Custom()


    # The combination of training_data, test_data, and the 
    # selected neural network architecture makes a problem instance of CORNN
    CORNN_benchmark_instance=CORNN.NN_Benchmark(training_data,test_data,neural_network_architecture)
    instance_dimension= CORNN_benchmark_instance.get_weight_count()

    
    example_candidate_solution=np.random.rand(instance_dimension)

    # In order to evaluate a candidate solution on the training set simply use:
    training_loss=CORNN_benchmark_instance.training_set_evaluation(example_candidate_solution)
    print("Training set loss:",training_loss)

    # In order to evaluate a candidate solution on the training set simply use:
    testing_loss=CORNN_benchmark_instance.testing_set_evaluation(example_candidate_solution)
    print("Testing set loss:",testing_loss)




