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
        self.resnet18 = models.resnet18(pretrained=True)
#         for param in self.resnet18.parameters():
#             param.requires_grad_(False)
            
        
        self.dropout = nn.Dropout(0.2)
        
        self.resnet18.fc = nn.Linear(512, 512)
        self.linear1 = nn.Linear(512, 256)
        self.linear2 = nn.Linear(256, 128)

        self.polynomials = nn.Linear(128, 2 * 6)
        
        for m in [self.resnet18.fc, self.linear1, self.linear2, self.polynomials]:
            if isinstance(m, nn.Linear):
                #I.xavier_normal_(m.weight)
                nn.init.normal_(m.weight, 0, 0.01)
                #nn.init.normal_(m.bais, 0, 0.01)
        
     
    def forward(self, x):
        x = self.resnet18(x)
        #x = x.view(x.shape[0], -1)
        #x = x.view(-1, 7 * 7 * 1280)
        
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.dropout(F.relu(self.linear2(x)))
        
        
        polynomials = self.polynomials(x)       
        
        return polynomials
        