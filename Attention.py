import torch
from torch import nn
from torch.nn.modules.activation import ReLU


class Senet(nn.Module):
    def __init__(self,channel,ratio=16):
        self.avg_pool=nn.AdaptiveAvgPool2d(1)#定义全局平均池化，输出尺度全为1
        self.fc      =nn.Sequential(
            nn.Linear(channel,channel//ratio,False),
            nn.ReLU(),
            nn.Linear(channel//ratio,channel,False),
            nn.Sigmoid(),
        )

    def foward(self,x):
        b,c,h,w=x.size()
        avg = self.avg_pool(x).view([b,c])
        fc  = self.fc(avg).view([b,c,1,1])

        return x*fc



class channel_attention(nn.Module):
    def __init__(self,channel,ratio=16):
        super(channel_attention, self).__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.fc     =nn.Sequential(
            nn.Linear(channel,channel//ratio,False),
            nn.ReLU(),
            nn.Linear(channel//ratio,channel,False),
        )
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        b,c,h,w=x.size()
        maxpool_out=self.maxpool(x).view([b,c])
        avgpool_out=self.avgpool(x).view([b,c])

        max_fc_out=self.fc(maxpool_out)
        avg_fc_out=self.fc(avgpool_out)

        out=max_fc_out+avg_fc_out
        out=self.sigmoid(out).view([b,c,1,1])

        return out*x


class spatial_attention(nn.Module):
    def __init__(self,kernel_size=7):
        super(spatial_attention, self).__init__()
        padding=7//2
        self.conv=nn.Conv2d(2,1,kernel_size,1,padding,bias=False)
        self.sigmoid=nn.Sigmoid()

    def foward(self,x):
        max_pool_out=torch.max(x,dim=1,keepdim=True)      #通道是在第一维度所以dim=1，就是通道进行最大化
        avg_pool_out = torch.mean(x, dim=1, keepdim=True)
        pool_out=torch.cat([max_pool_out,avg_pool_out],dim=1) #把两个粘合在一起
        out=self.conv(pool_out)
        out = self.sigmoid(out)
        return out*x



class cbam(nn.Module):
    def __init__(self,channel,ratio=16,kernel_size=7):
        super(cbam, self).__init__()
        self.channel_attention = channel_attention(channel,ratio)
        self.spatial_attention = spatial_attention(kernel_size)
    def forward(self,x):
        x= self.channel_attention(x)
        x=self.spatial_attention(x)
        return  x