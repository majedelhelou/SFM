import torch
import torch.nn as nn

EPS = 1e-3

class ResBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                     padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                     padding=1, bias=True)
    
    def forward(self, x):
        residual = x
        
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        
        out += residual
        out = self.relu(out)
        
        return out
        
class ResBlockNet(nn.Module):
    def __init__(self):
        super(ResBlockNet, self).__init__()
        self.resblock_layer = self.make_layer(16)
        
        channels = 3
        
        self.input = nn.Conv2d(in_channels=channels, out_channels=64,
                kernel_size=3, stride=1, padding=1, bias=False)
        
        self.output = nn.Conv2d(in_channels=64, out_channels=channels,
                kernel_size=3, stride=1, padding=1, bias=False)
        
        self.relu = nn.ReLU(inplace=True)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                #nn.init.kaiming_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                
    def make_layer(self, count):
        layers = []
        for _ in range(count):
            layers.append(ResBlock(64, 64))
        return nn.Sequential(*layers)

    def forward(self, x2):
        residual = x2
        out = self.relu(self.input(x2))
        out = self.resblock_layer(out)
        out = self.output(out)
        # out = torch.add(out,residual)
        return out