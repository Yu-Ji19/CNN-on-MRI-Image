import torch
import numpy as np
import torch.nn as nn
from utils import *
from layers import *

class down_block(nn.Module):
    """
    downsampling block in spherical unet
    mean pooling => (conv => BN => ReLU) * 2
    
    """
    def __init__(self, conv_layer, in_ch, out_ch, neigh_orders, pool_neigh_orders, first = False):
        super(down_block, self).__init__()


#        Batch norm version
        if first:
            self.block = nn.Sequential(
                conv_layer(in_ch, out_ch, neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True),
                conv_layer(out_ch, out_ch, neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True)
        )
            
        else:
            self.block = nn.Sequential(
                pool_layer(pool_neigh_orders, 'mean'),
                conv_layer(in_ch, out_ch, neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True),
                conv_layer(out_ch, out_ch, neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True),
        )


    def forward(self, x):
        # batch norm version
        x = self.block(x)
        
        return x


class up_block(nn.Module):
    """Define the upsamping block in spherica uent
    upconv => (conv => BN => ReLU) * 2
    
    Parameters:
            in_ch (int) - - input features/channels
            out_ch (int) - - output features/channels    
            neigh_orders (tensor, int)  - - conv layer's filters' neighborhood orders
            
    """    
    def __init__(self, conv_layer, in_ch, out_ch, neigh_orders, upconv_top_index, upconv_down_index):
        super(up_block, self).__init__()
        
        self.up = upconv_layer(in_ch, out_ch, upconv_top_index, upconv_down_index)
        
        # batch norm version
        self.double_conv = nn.Sequential(
             conv_layer(in_ch, out_ch, neigh_orders),
             nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
             nn.LeakyReLU(0.2, inplace=True),
             conv_layer(out_ch, out_ch, neigh_orders),
             nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
             nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x1, x2):
        
        x1 = self.up(x1)
        x = torch.cat((x1, x2), 1) 
        x = self.double_conv(x)

        return x

class vgg16(nn.Module):
    def __init__(self, in_ch):
        super(vgg16, self).__init__()
     
        neigh_orders = Get_neighs_order()[2:]
        
        chs = [in_ch, 32, 64, 128, 256, 512, 1024]
        conv_layer = onering_conv_layer

        sequence = []
        sequence.append(conv_layer(chs[0], chs[1], neigh_orders[0]))
        sequence.append(nn.BatchNorm1d(chs[1]))
        sequence.append(nn.LeakyReLU(0.1, inplace=True))
        sequence.append(conv_layer(chs[1], chs[1], neigh_orders[0]))
        sequence.append(nn.BatchNorm1d(chs[1]))
        sequence.append(nn.LeakyReLU(0.1, inplace=True))
        sequence.append(conv_layer(chs[1], chs[1], neigh_orders[0]))
        sequence.append(nn.BatchNorm1d(chs[1]))
        sequence.append(nn.LeakyReLU(0.1, inplace=True))
            
        for l in range(1, len(chs)-1):
            sequence.append(pool_layer(neigh_orders[l-1], 'mean'))
            sequence.append(conv_layer(chs[l], chs[l+1], neigh_orders[l]))
            sequence.append(nn.BatchNorm1d(chs[l+1]))
            sequence.append(nn.LeakyReLU(0.1, inplace=True))
            sequence.append(conv_layer(chs[l+1], chs[l+1], neigh_orders[l]))
            sequence.append(nn.BatchNorm1d(chs[l+1]))
            sequence.append(nn.LeakyReLU(0.1, inplace=True))

        self.model = nn.Sequential(*sequence)    
        

    def forward(self, x):
        x = self.model(x)
        x = torch.mean(x, 0, True)
        return x

class Multitask_vgg16_Final_Regression(nn.Module):
    def __init__(self, in_ch):
        super(Multitask_vgg16_Final_Regression, self).__init__()
        self.in_ch = in_ch

        self.left_feature = vgg16(in_ch)
        self.right_feature = vgg16(in_ch)
    
        self.regression = nn.Sequential(
            nn.Linear(2048, 256),
            nn.Linear(256, 64),
            nn.Linear(64, 1),
        )
        self.output_gender = nn.Sequential(
            nn.Linear(2048, 256),
            nn.Linear(256, 64),
            nn.Linear(64, 2),
            nn.Softmax()
        )

    def forward(self, x):
        left = x[:, :self.in_ch]
        right = x[:, self.in_ch:]
        left_output = self.left_feature(left)
        right_output = self.right_feature(right)
        left_output = torch.squeeze(left_output, 0)
        right_output = torch.squeeze(right_output, 0)

        feature = torch.cat((left_output, right_output))

        age = self.regression(feature)
    
        gender = self.output_gender(feature).float()
        

        y = torch.cat((age, gender))
        return y


class vgg16_Dropout(nn.Module):
    def __init__(self, in_ch, dropout_p=0.1):
        super(vgg16_Dropout, self).__init__()
     
        neigh_orders = Get_neighs_order()[2:]
        
        chs = [in_ch, 32, 64, 128, 256, 512, 1024]
        conv_layer = onering_conv_layer

        sequence = []
        sequence.append(conv_layer(chs[0], chs[1], neigh_orders[0]))
        sequence.append(nn.BatchNorm1d(chs[1]))
        sequence.append(nn.LeakyReLU(0.1, inplace=True))
        sequence.append(nn.Dropout(dropout_p))
        sequence.append(conv_layer(chs[1], chs[1], neigh_orders[0]))
        sequence.append(nn.BatchNorm1d(chs[1]))
        sequence.append(nn.LeakyReLU(0.1, inplace=True))
        sequence.append(nn.Dropout(dropout_p))       
        sequence.append(conv_layer(chs[1], chs[1], neigh_orders[0]))
        sequence.append(nn.BatchNorm1d(chs[1]))
        sequence.append(nn.LeakyReLU(0.1, inplace=True))
        sequence.append(nn.Dropout(dropout_p))            
        for l in range(1, len(chs)-1):
            sequence.append(pool_layer(neigh_orders[l-1], 'mean'))
            sequence.append(conv_layer(chs[l], chs[l+1], neigh_orders[l]))
            sequence.append(nn.BatchNorm1d(chs[l+1]))
            sequence.append(nn.LeakyReLU(0.1, inplace=True))
            sequence.append(nn.Dropout(dropout_p))
            sequence.append(conv_layer(chs[l+1], chs[l+1], neigh_orders[l]))
            sequence.append(nn.BatchNorm1d(chs[l+1]))
            sequence.append(nn.LeakyReLU(0.1, inplace=True))
            sequence.append(nn.Dropout(dropout_p))

        self.model = nn.Sequential(*sequence)    
        self.fc =  nn.Sequential(
                nn.Linear(chs[-1], 1)
                )

    def forward(self, x):
        x = self.model(x)
        x = torch.mean(x, 0, True)
        x = self.fc(x)
        return x




class res_block(nn.Module):
    def __init__(self, c_in, c_out, neigh_orders, first_in_block=False):
        super(res_block, self).__init__()
        
        self.conv1 = onering_conv_layer(c_in, c_out, neigh_orders)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.relu = nn.LeakyReLU(0.2)
        self.conv2 = onering_conv_layer(c_out, c_out, neigh_orders)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.first = first_in_block
    
    def forward(self, x):
        res = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        if self.first:
            res = torch.cat((res,res),1)
        x = x + res
        x = self.relu(x)
        
        return x

class ResNet(nn.Module):
    def __init__(self, in_c):
        super(ResNet, self).__init__()
        neigh_orders = Get_neighs_order()[2:]
        self.in_ch = in_c
        
        self.conv1 =  onering_conv_layer(in_c, 64, neigh_orders[0])
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.LeakyReLU(0.2)
        
        self.pool1 = pool_layer(neigh_orders[1], 'max')
        self.res1_1 = res_block(64, 128, neigh_orders[1], True)
        self.res1_2 = res_block(128, 128, neigh_orders[1])
        self.res1_3 = res_block(128, 128, neigh_orders[1])
        
        self.pool2 =pool_layer(neigh_orders[2], 'max')
        self.res2_1 = res_block(128, 256, neigh_orders[2], True)
        self.res2_2 = res_block(256, 256, neigh_orders[2])
        self.res2_3 = res_block(256, 256, neigh_orders[2])
        
        self.pool3 = pool_layer(neigh_orders[3], 'max')
        self.res3_1 = res_block(256, 512, neigh_orders[3], True)
        self.res3_2 = res_block(512, 512, neigh_orders[3])
        self.res3_3 = res_block(512, 512, neigh_orders[3])
                
        self.pool4 = pool_layer(neigh_orders[4], 'max')
        self.res4_1 = res_block(512, 1024, neigh_orders[4], True)
        self.res4_2 = res_block(1024, 1024, neigh_orders[4])
        self.res4_3 = res_block(1024, 1024, neigh_orders[4])
        
        self.outc = nn.Sequential(
                nn.Linear(1024, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 64),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.pool1(x)[0]
        x = self.res1_1(x)
        x = self.res1_2(x)
        x = self.res1_3(x)
        
        x = self.pool2(x)[0]
        x = self.res2_1(x)
        x = self.res2_2(x)
        x = self.res2_3(x)
        
        x = self.pool3(x)[0]
        x = self.res3_1(x)
        x = self.res3_2(x)
        x = self.res3_3(x)
                
        x = self.pool4(x)[0]
        x = self.res4_1(x)
        x = self.res4_2(x)
        x = self.res4_3(x)
    
        
        x = torch.mean(x, 0, True)
        x = self.outc(x)
        return x


class ResNet_Final_Regression(nn.Module):
    def __init__(self, in_c):
        super(ResNet_Final_Regression, self).__init__()
        # _, _, neigh_orders_10242, neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42, _ = Get_neighs_order()
        neigh_orders = Get_neighs_order()[2:]
        self.in_ch = in_c
        self.left_net = ResNet(in_c)
        self.right_net = ResNet(in_c)
        self.output = nn.Sequential(
            nn.Linear(2, 1)
        )

    def forward(self, x):
        left = x[:, :self.in_ch]
        right = x[:, self.in_ch:]
        left_output = self.left_net(left)
        right_output = self.left_net(right)
        
        left_output = torch.squeeze(left_output, 0)
        right_output = torch.squeeze(right_output, 0)

        y = self.output(torch.cat((left_output, right_output)))
        y = torch.squeeze(y)
        return y

class Multitask_ResNet_Final_Regression(nn.Module):
    def __init__(self, in_c):
        super(Multitask_ResNet_Final_Regression, self).__init__()
        # _, _, neigh_orders_10242, neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42, _ = Get_neighs_order()
        neigh_orders = Get_neighs_order()[2:]
        self.in_ch = in_c
        self.left_net = ResNet(in_c)
        self.right_net = ResNet(in_c)
        self.regression = nn.Sequential(
            nn.Linear(2, 3)
        )
        self.output_gender = nn.Sequential(
            nn.Softmax()
        )

    def forward(self, x):
        left = x[:, :self.in_ch]
        right = x[:, self.in_ch:]
        left_output = self.left_net(left)
        right_output = self.left_net(right)
        
        left_output = torch.squeeze(left_output, 0)
        right_output = torch.squeeze(right_output, 0)

        regression = self.regression(torch.cat((left_output, right_output)))
        output_age = regression[0]
        output_gender = self.output_gender(regression[1:3]).float()

        y = torch.cat((output_age.unsqueeze(0), output_gender))

        return y


class dense_block(nn.Module):
    def __init__(self, ch, neigh_orders):
        super(dense_block, self).__init__()
        
        self.conv1 = nn.Sequential(
                nn.BatchNorm1d(ch),
                nn.LeakyReLU(0.2, inplace=True),
                onering_conv_layer(ch, ch, neigh_orders)
                )

        self.conv2 = nn.Sequential(
                nn.BatchNorm1d(ch*2),
                nn.LeakyReLU(0.2, inplace=True),
                onering_conv_layer(ch*2, ch, neigh_orders)
                )
        self.conv3 = nn.Sequential(
                nn.BatchNorm1d(ch*3),
                nn.LeakyReLU(0.2, inplace=True),
                onering_conv_layer(ch*3, ch, neigh_orders)
              
                )
        self.conv4 = nn.Sequential(
                nn.BatchNorm1d(ch*4),
                nn.LeakyReLU(0.2, inplace=True),
                onering_conv_layer(ch*4, ch, neigh_orders)
                )
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x,x1), 1))
        x3 = self.conv3(torch.cat((x,x1,x2), 1))
        x4 = self.conv4(torch.cat((x,x1,x2,x3), 1))
        return x4


class DenseNet(nn.Module):
    def __init__(self, in_ch):
        super(DenseNet, self).__init__()
     
        _, _, neigh_orders_10242, neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42, neigh_orders_12 = Get_neighs_order()
       
        self.model = nn.Sequential(       
                        onering_conv_layer(in_ch, 48, neigh_orders_10242),
                        dense_block(48, neigh_orders_10242),
                        pool_layer(neigh_orders_10242, 'mean'),
                        dense_block(48, neigh_orders_2562),
                        pool_layer(neigh_orders_2562, 'mean'),
                        dense_block(48, neigh_orders_642),
                        pool_layer(neigh_orders_642, 'mean'),
                        dense_block(48, neigh_orders_162),
                        pool_layer(neigh_orders_162, 'mean'),
                        dense_block(48, neigh_orders_42),
                        pool_layer(neigh_orders_42, 'mean')
        )

        self.fc = nn.Linear(12*48, 1)
        self.out = nn.Sigmoid()
        
    def forward(self, x):
        x = self.model(x)
        x = x.view(1, x.size()[0] * x.size()[1])
        x = self.out(self.fc(x))
        
        return x


