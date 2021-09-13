import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# If you wish to utile a different NN architecture, then all that
# is required is you implement a custom version in line with the 
# model classes below.
# 
# The model class have been intentionally been left in a verbose 
# form to facilitate ease of interpretation for those not
# necessarily familiar with PyTorch. 


class Net_5_relu_layers(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 10)     # input->H1
        self.fc2 = nn.Linear(10, 10)    # H1->H2
        self.fc3 = nn.Linear(10, 10)    # H2->H3
        self.fc4 = nn.Linear(10, 10)    # H3->H4
        self.fc5 = nn.Linear(10, 10)    # H4->H5
        self.fc6 = nn.Linear(10, 1)     # H5->output

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = F.relu(x)
        x = self.fc6(x)
        return x

class Net_5_tanh_layers(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 10)     # input->H1
        self.fc2 = nn.Linear(10, 10)    # H1->H2
        self.fc3 = nn.Linear(10, 10)    # H2->H3
        self.fc4 = nn.Linear(10, 10)    # H3->H4
        self.fc5 = nn.Linear(10, 10)    # H4->H5
        self.fc6 = nn.Linear(10, 1)     # H5->output

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.fc3(x)
        x = torch.tanh(x)
        x = self.fc4(x)
        x = torch.tanh(x)
        x = self.fc5(x)
        x = torch.tanh(x)
        x = self.fc6(x)
        return x

class Net_3_relu_layers(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 10)     # input->H1
        self.fc2 = nn.Linear(10, 10)    # H1->H2
        self.fc3 = nn.Linear(10, 10)    # H2->H3
        self.fc4 = nn.Linear(10, 1)     # H3->output

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return x

class Net_3_tanh_layers(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 10)     # input->H1
        self.fc2 = nn.Linear(10, 10)    # H1->H2
        self.fc3 = nn.Linear(10, 10)    # H2->H3
        self.fc4 = nn.Linear(10, 1)     # H3->output

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.fc3(x)
        x = torch.tanh(x)
        x = self.fc4(x)
        return x


class Net_1_relu_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 10)     # input->H1
        self.fc2 = nn.Linear(10, 1)     # H1->output
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
        
class Net_1_tanh_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 10)     # input->H1
        self.fc2 = nn.Linear(10, 1)     # H1->output
    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        return x



class NN_Benchmark:
    def __init__(self,training_set,test_set,network):
        self.training_set=training_set
        self.test_set=test_set
        self.network=network
        self.loss_function=nn.MSELoss()
    def training_set_evaluation(self,_weights):
        _weights=torch.from_numpy(_weights.astype(np.float32))
        new_state_dict=unflatten_state_dict(self.network.state_dict(),_weights)
        
        with torch.no_grad():
            self.network.load_state_dict(new_state_dict)
            self.network.eval()

            patterns, targets=self.training_set
            #use a single forward pass for all the data
            predictions=self.network(patterns.view(-1,2))
            loss=self.loss_function(predictions,targets)
            return float(loss)
    
    def testing_set_evaluation(self,_weights):
        _weights=torch.from_numpy(_weights.astype(np.float32))
        new_state_dict=unflatten_state_dict(self.network.state_dict(),_weights)
        
        with torch.no_grad():
            self.network.load_state_dict(new_state_dict)
            self.network.eval()

            patterns, targets=self.test_set
            #use a single forward pass for all the data
            predictions=self.network(patterns.view(-1,2))
            loss=self.loss_function(predictions,targets)
            return float(loss)

    def evaluate_training_set(self):
        with torch.no_grad():
            self.network.eval()
            patterns, targets=self.training_set
            #uses a single forward pass for all the data
            predictions=self.network(patterns.view(-1,2))
            loss=self.loss_function(predictions,targets)
            return loss
    def evaluate_testing_set(self):
        with torch.no_grad():
            self.network.eval()
            patterns, targets=self.test_set
            #uses a single forward pass for all the data
            predictions=self.network(patterns.view(-1,2))
            loss=self.loss_function(predictions,targets)
            return loss

    def inference_training_set(self):
        with torch.no_grad():
            self.network.eval()
            patterns, targets=self.training_set
            predictions=self.network(patterns.view(-1,2))
            return predictions,targets

    def get_flatten_weights(self):
        return list(self.network.parameters())

    def get_weight_count(self):
        weights=self.network.state_dict()
        size=flatten_state_dict(weights).size()[0]
        return size

    # full_grad_epoch member function here left as a convenience to a user
    # as it can be used as an Adam based baseline
    def full_grad_epoch(self):
        self.network.train()
        self.network.zero_grad()
        optimizer = optim.Adam(self.network.parameters(), lr=0.001)
        patterns, targets=self.training_set
        predictions=self.network(patterns.view(-1,2))
        loss=self.loss_function(predictions,targets)
        loss.backward()
        optimizer.step()


from collections import OrderedDict 
def flatten_state_dict(_pytorch_state_dict):
    long_tensor=torch.empty(0)
    for key,tensor in _pytorch_state_dict.items():
        long_tensor=torch.cat((long_tensor, tensor.flatten()), 0)
    return long_tensor


def unflatten_state_dict(_example_pytorch_state_dict,_flat_tensor):
    new_state=OrderedDict()
    current_index=0
    for key,tensor in _example_pytorch_state_dict.items():
        current_shape=tensor.size()
        number_of_elements_to_copy=tensor.numel()
        current_tensor=_flat_tensor[current_index:current_index+number_of_elements_to_copy]
        current_index=current_index+number_of_elements_to_copy
        current_tensor=current_tensor.view(current_shape)
        new_state[key]=current_tensor
    return new_state
