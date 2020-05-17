import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

class PolyNet(nn.Module):
    
    def __init__(self, num_lanes=5):
        super(PolyNet, self).__init__()
        # Use a pre-trained network for feature extraction
        # Checkout https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py
        mobilenetv2 = models.mobilenet_v2(pretrained=True)
        for param in mobilenetv2.parameters():
            param.requires_grad_(False)
            
        self.feature_extractor = mobilenetv2.features
        
        self.dropout = nn.Dropout(0.2)
        
        self.linear1 = nn.Linear(mobilenetv2.last_channel * 7 * 7, 512)
        self.linear2 = nn.Linear(512, 128)
        
        self.existence = nn.Linear(128, num_lanes)
        self.polynomials = nn.Linear(128, num_lanes * 6)
        
     
    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.shape[0], -1)
        #x = x.view(-1, 7 * 7 * 1280)
        
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.dropout(F.relu(self.linear2(x)))
        
        objectness = F.sigmoid(self.existence(x)) # objectness probablity lies in [0,1]
        polynomials = self.polynomials(x)
        #bounds = F.sigmoid(self.bounds(x)) # bounds lie within [0,1]
        
        return (objectness, polynomials)
        
