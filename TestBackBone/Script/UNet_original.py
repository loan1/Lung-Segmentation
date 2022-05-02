#https://github.com/IlliaOvcharenko/lung-segmentation/blob/master/src/models.py
import torch
import torchvision

import pandas as pd
import numpy as np
from torchsummary import summary

class Block(torch.nn.Module):
    def __init__(self, in_channels, mid_channel, out_channels):
        super().__init__()
        
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=mid_channel, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
       
        x = torch.nn.functional.relu(x, inplace=True)
        
        x = self.conv2(x)
     
        out = torch.nn.functional.relu(x, inplace=True)
        return out
    

class UNet_original123(torch.nn.Module):
    def up(self, x, size):
        return torch.nn.functional.interpolate(x, size=size, mode=self.upscale_mode)
    
    def down(self, x):
        return torch.nn.functional.max_pool2d(x, kernel_size=2)
    
    def __init__(self, in_channels, out_channels, upscale_mode="nearest"):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upscale_mode = upscale_mode
        
        self.enc1 = Block(in_channels, 64, 64)
        self.enc2 = Block(64, 128, 128)
        self.enc3 = Block(128, 256, 256 )
        self.enc4 = Block(256, 512, 512)
        
        self.center = Block(512, 1024, 512)
        
        self.dec4 = Block(1024, 512, 256)
        self.dec3 = Block(512, 256, 128)
        self.dec2 = Block(256, 128, 64)
        self.dec1 = Block(128, 64, 64)
        
        self.out = torch.nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.down(enc1))
        enc3 = self.enc3(self.down(enc2))
        enc4 = self.enc4(self.down(enc3))
        
        center = self.center(self.down(enc4))
        
        dec4 = self.dec4(torch.cat([self.up(center, enc4.size()[-2:]), enc4], 1))
        dec3 = self.dec3(torch.cat([self.up(dec4, enc3.size()[-2:]), enc3], 1))
        dec2 = self.dec2(torch.cat([self.up(dec3, enc2.size()[-2:]), enc2], 1))
        dec1 = self.dec1(torch.cat([self.up(dec2, enc1.size()[-2:]), enc1], 1))
        
        out = self.out(dec1)
        
        return out  


# model = UNet_original123(in_channels=1, out_channels=1)
# model = model.cuda()
# summary(model, (1,256,256))
