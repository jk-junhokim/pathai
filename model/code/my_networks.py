import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch import Tensor
import numpy as np


##################################################
################ BUILDING BLOCKS #################
##################################################
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001,
                                 momentum=0.1,
                                 affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    

class BlockA(nn.Module):
    def __init__(self, scale=1.0):
        super(BlockA, self).__init__()

        self.scale = scale
        
        self.branch0 = BasicConv2d(320, 32, kernel_size=1, stride=1)
        
        self.branch1 = nn.Sequential(
            BasicConv2d(320, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )
        
        self.branch2 = nn.Sequential(
            BasicConv2d(320, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 48, kernel_size=3, stride=1, padding=1),
            BasicConv2d(48, 64, kernel_size=3, stride=1, padding=1)
        )

        self.conv2d = nn.Conv2d(128, 320, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        
        return out
    
    
class BlockB(nn.Module):

    def __init__(self, scale=1.0):
        super(BlockB, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(1088, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(1088, 128, kernel_size=1, stride=1),
            BasicConv2d(128, 160, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(160, 192, kernel_size=(7,1), stride=1, padding=(3,0))
        )

        self.conv2d = nn.Conv2d(384, 1088, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        
        return out
    

class BlockM(nn.Module):

    def __init__(self):
        super(BlockM, self).__init__()

        self.branch0 = BasicConv2d(192, 96, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(192, 48, kernel_size=1, stride=1),
            BasicConv2d(48, 64, kernel_size=5, stride=1, padding=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(192, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1),
            BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(192, 64, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out   
    

class ReductionA(nn.Module):

    def __init__(self):
        super(ReductionA, self).__init__()

        self.branch0 = BasicConv2d(320, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv2d(320, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        
        return out

##################################################
################### BASE MODULE ##################
##################################################
class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def name(self):
        return self._name

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


##################################################
################# VGGINCEPTION ###################
##################################################
class VggInception(BaseNetwork):
    
    def __init__(self, in_channels, num_classes):
        super(VggInception, self).__init__()

        # * VGG Layer
        self.vgg_features = nn.Sequential(
            # block 1
            BasicConv2d(in_planes=in_channels, out_planes=64, kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_planes=64, out_planes=64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # block 2
            BasicConv2d(in_planes=64, out_planes=128, kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_planes=128, out_planes=128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # block 3
            BasicConv2d(in_planes=128, out_planes=256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_planes=256, out_planes=256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_planes=256, out_planes=256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # block 4
            BasicConv2d(in_planes=256, out_planes=512, kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_planes=512, out_planes=512, kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_planes=512, out_planes=512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # * InceptionV4 (Inception-A) Layer
        # Average Pool & 1x1 conv
        self.inception_b1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(in_planes=512, out_planes=128, kernel_size=1, stride=1)
        )

        # 1x1 conv
        self.inception_b2 = nn.Sequential(
            BasicConv2d(in_planes=512, out_planes=96, kernel_size=1, stride=1)
        )

        # 1x1 & 3x3 conv
        self.inception_b3 = nn.Sequential(
            BasicConv2d(in_planes=512, out_planes=64, kernel_size=1, stride=1),
            BasicConv2d(in_planes=64, out_planes=128, kernel_size=3, stride=1, padding=1)
        )

        # 1x1 & 3x3 & 3x3 conv
        self.inception_b4 = nn.Sequential(
            BasicConv2d(in_planes=512, out_planes=64, kernel_size=1, stride=1),
            BasicConv2d(in_planes=64, out_planes=96, kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_planes=96, out_planes=96, kernel_size=3, stride=1, padding=1)
        )
        
        self.max = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier_n1 = nn.Sequential(
            nn.Linear(7168, 4096) # input size = 128 x 128
            # nn.Linear(28672, 4096) # input size = 256 x 256
            # nn.Linear(64512, 4096) # input size = 384 x 384
            # nn.Linear(114688, 4096) # input size = 512 x 512
        )
        
        self.classifier_n2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.Linear(4096, 1024),
            nn.Linear(1024, num_classes)
        )
        
        self.initialize_weights()    
    
    def forward(self, x):
        # pass through vgg
        x = self.vgg_features(x)

        # pass through inception
        branch1 = self.inception_b1(x)
        branch2 = self.inception_b2(x)
        branch3 = self.inception_b3(x)
        branch4 = self.inception_b4(x)
        outputs = [branch1, branch2, branch3, branch4]
        x = torch.cat(outputs, 1)
        x = self.max(x)

        # classify
        x = torch.flatten(x, 1)
        x = F.dropout(x, p=0.4)
        x = self.classifier_n1(x)
        x = self.classifier_n2(x)
        
        return x
        

##################################################
################ InceptionResNet #################
##################################################
class InceptionResNet(BaseNetwork):
    def __init__(self, in_channels, num_classes):
        super(InceptionResNet, self).__init__()
        
        self.stem = nn.Sequential(
            BasicConv2d(in_planes=in_channels, out_planes=32, kernel_size=3, stride=1),
            BasicConv2d(in_planes=32, out_planes=32, kernel_size=3, stride=1),
            BasicConv2d(in_planes=32, out_planes=64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(3, stride=2),
            BasicConv2d(64, 80, kernel_size=1, stride=1),
            BasicConv2d(80, 192, kernel_size=3, stride=1),
            nn.MaxPool2d(3, stride=2) 
        )
        
        self.mix = nn.Sequential(
            BlockM()
        )
        
        self.blockA_repeat = nn.Sequential(
            BlockA(scale=0.17),
            BlockA(scale=0.17),
            BlockA(scale=0.17),
            BlockA(scale=0.17),
            BlockA(scale=0.17),
            BlockA(scale=0.17),
            BlockA(scale=0.17)
        )
        
        self.reduction_A = ReductionA()
        
        self.blockB_repeat = nn.Sequential(
            BlockB(scale=0.1),
            BlockB(scale=0.1),
            BlockB(scale=0.1),
            BlockB(scale=0.1),
            BlockB(scale=0.1),
            BlockB(scale=0.1),
            BlockB(scale=0.1),
            BlockB(scale=0.1),
            BlockB(scale=0.1),
            BlockB(scale=0.1),
            BlockB(scale=0.1),
            BlockB(scale=0.1),
            BlockB(scale=0.1),
            BlockB(scale=0.1)
        )
        
        self.basic_conv = BasicConv2d(1088, 768, kernel_size=1, stride=1)
        self.avg_pool = nn.AvgPool2d(2, count_include_pad=False)
        
        self.last_linear = nn.Sequential(
            nn.Linear(37632, num_classes)
        )
        
    def forward(self, x):
        # torch.Size([256, 3, 128, 128])
        x = self.stem(x) # torch.Size([256, 192, 29, 29])
        x = self.mix(x) # torch.Size([256, 320, 29, 29])
        x = self.blockA_repeat(x) # torch.Size([256, 320, 29, 29])
        x = self.reduction_A(x) # torch.Size([256, 1088, 14, 14])
        x = self.blockB_repeat(x) # torch.Size([256, 1088, 14, 14])
        x = self.basic_conv(x) # torch.Size([256, 768, 14, 14])
        x = self.avg_pool(x) # torch.Size([256, 768, 7, 7])
        
        
        x = torch.flatten(x, 1) # torch.Size([256, 37632])
        x = F.dropout(x, p=0.4) # torch.Size([256, 37632])
        x = self.last_linear(x) # torch.Size([256, 2])

        return x
    
    
##################################################
################# PRETRAINED MODELS ##############
##################################################


