# CORNN - Continuous Optimisation of Regression tasks using Neural Networks

CORNN is easy to native python 3 benchmark suit. 

The benchmark suit consists of 54 classic real-valued functions from the field of continuous optimization, from which regression data sets have been generated by uniformly sampling the active domains each function. For each function a training set and test set of 3750 and 1250 are provided.
A CORNN benchmark instance is generated by pairing a function’s data set (training and test) with specific neural network architecture. In the case of CORNN there are 6 architecture provided to you. The goal on each benchmark is to try and improve the model fit by optimization of the NN weights.

## Features

## Code illustration of how to use CORNN
```py
function_dictionary=CORNN.get_benchmark_functions()
# Returns a dictionary of all the regression function functions within CORNN
# The key is the function name, the value is a 3 element tuple:
# Containing the raw objective function (for example the Ackley function)
# The X variable's domain and the Y variable's domain

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

neural_network_architecture=neural_network_dictionary["Net_5_relu_layers"]() # the () to instantiate 
# Selects the 3 hiddern layer NN model that uses ReLU activation function within
# the Hiddern layers. 


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
```
This code is also present in Demo.py

## Underlying Functions
The CORNN suite provides data sets for the following functions:
...
