import lib.Benchmark_Functions_2D_Definition as Fd
import lib.NN_Benchmarking as NN_Benchmarking# this is just for reflection
from lib.NN_Benchmarking import *
import inspect
import csv
import numpy as np
import pandas as pd
import torch


DATA_PATH="lib/data_sets/"
def get_function_data_csv(_file_dir):
    file = pd.read_csv(_file_dir)
    result = file.values
    return result

def get_benchmark_functions():
    raw_functions_definitions = inspect.getmembers(Fd, inspect.isfunction)
    function_dictionary={}
    for function_definition in raw_functions_definitions:
        function_dictionary[function_definition[1]()[Fd.FUNCTION_NAME]]=function_definition[1]()
    return function_dictionary

def get_scaled_function_data(_function):
    function_name=_function[Fd.FUNCTION_NAME]
    x_ranges=_function[Fd.FUNCTION_XRANGE]
    y_ranges=_function[Fd.FUNCTION_YRANGE]
    training_data_file=DATA_PATH+function_name+"_train_set.csv"  
    testing_data_file=DATA_PATH+function_name+"_test_set.csv"

    #Process training data and testing data
    
    train_data=get_function_data_csv(training_data_file)
    #scale all input within [-1,1] based on known function domain
    x_s=train_data[:,0]
    x_s=2*(x_s-x_ranges[0])/(x_ranges[1]-x_ranges[0])-1
    y_s=train_data[:,1]
    y_s=2*(y_s-y_ranges[0])/(y_ranges[1]-y_ranges[0])-1
    train_patterns=np.array(list(zip(x_s,y_s)))

    #scale all outputs within [-1,1] using min and max values from training set
    train_labels=train_data[:,[2]]
    max_label=np.amax(train_labels)
    min_label=np.amin(train_labels)
    train_labels=2*(train_labels-min_label)/(max_label-min_label)-1


    test_data=get_function_data_csv(testing_data_file)
    #scale all input within [-1,1] based on known function domain
    x_s=test_data[:,0]
    x_s=2*(x_s-x_ranges[0])/(x_ranges[1]-x_ranges[0])-1
    y_s=test_data[:,1]
    y_s=2*(y_s-y_ranges[0])/(y_ranges[1]-y_ranges[0])-1
    test_patterns=np.array(list(zip(x_s,y_s)))
    #scale all outputs within [-1,1] using min and max values from training set not test set.
    test_labels=test_data[:,[2]]
    test_labels=2*(test_labels-min_label)/(max_label-min_label)-1

    #Convert to PyTorch Tensors
    train_patterns_tensor=torch.from_numpy(train_patterns.astype(np.float32))
    train_labels_tensor=torch.from_numpy(train_labels.astype(np.float32))
    test_patterns_tensor=torch.from_numpy(test_patterns.astype(np.float32))
    test_labels_tensor=torch.from_numpy(test_labels.astype(np.float32))

    return [train_patterns_tensor,train_labels_tensor], [test_patterns_tensor,test_labels_tensor]

def get_NN_models():
    full_class_list = inspect.getmembers(NN_Benchmarking, inspect.isclass)

    # Internal convention is that models start with “Net”,
    # this should be maintained to facilitate reflection
    NN_models={}
    for current_class in full_class_list:
        if(current_class[0][:3]=="Net"):
            NN_models[current_class[0]]=current_class[1]


    return NN_models