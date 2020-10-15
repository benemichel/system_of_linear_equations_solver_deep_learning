import torch
import torch.nn as nn

class Cnn(nn.Module):
    """
    custom model for math symbol labeling
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        '''Defines layers of a neural network.
        params:
           input_dim: Input layer dimensions (features)
           hidden_dim: Hidden layer dimensions
           output_dim: Output dimensions
         '''
        super(Cnn, self).__init__()   
           
        self.fc1 = nn.Linear(input_dim, hidden_dim)  #batch_size, channels*height*width
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        self.hidden_activation = nn.functional.relu()
        
        
    def forward(self, out):
        '''Feedforward sequence
        params:
            out: input data which will be transformed to model output
         '''
        
        out = out.view(out.size(0),-1)
        out = self.fc1(out)
        out = self.hidden_activation(self.fc2(out))
        out = self.fc3(out)
        
        return out