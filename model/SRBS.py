import torch
import torch.nn as nn
import torchvision.ops

class SRBS_high(nn.Module):
    def __init__(self,low_channels,high_channels,c_kernel=3,r_kernel=3,use_att=False,use_process=True):

        super(SRBS_high, self).__init__()

        self.l_c = low_channels
        self.h_c = high_channels
        self.c_k = c_kernel
        self.r_k = r_kernel
        self.att = use_att
        self.non_local_att = nn.Conv2d
        # self.dcn = torchvision.ops.deform_conv2d()
        if self.l_c == self.h_c:
            print('Channel checked!')
        else:
            raise ValueError('Low and Hih channels need to be the same!')
        # self.dcn = deform_conv2d(self.l_c,self.h_c,stride=1,padding=(0,self.r_k//2))
        self.dcn = nn.Conv2d(self.l_c,self.h_c,kernel_size=(3,3),stride=1,padding=(self.c_k//2,self.c_k//2))
        self.sigmoid = nn.Sigmoid()
        if self.att == True:
            self.csa = self.non_local_att(self.l_c,self.h_c,1,1,0)
        else:
            self.csa = None
        if use_process == True:
            self.preprocess = nn.Sequential(nn.Conv2d(self.l_c,self.h_c//2,1,1,0),nn.Conv2d(self.h_c//2,self.l_c,1,1,0))
        else:
            self.preprocess = None
    def forward(self,a_low,a_high):
        if self.preprocess is not None:
            a_low = a_low
            a_high = self.preprocess(a_high)
        else:
            a_low = a_low
            a_high = a_high

        a_low_c = self.dcn(a_low)
        a_low_cw = self.sigmoid(a_low_c)
        a_low_cw = a_low_cw * a_high
        a_colum = a_low + a_low_cw

        if self.csa is not None:
            a_TTOA = self.csa(a_colum)
        else:
            a_TTOA = a_colum
        return a_TTOA


class SRBS(nn.Module):
    def __init__(self, low_channels, high_channels, c_kernel=3, r_kernel=3, use_att=False, use_process=True):

        super(SRBS, self).__init__()

        self.l_c = low_channels
        self.h_c = high_channels
        self.c_k = c_kernel
        self.r_k = r_kernel
        self.att = use_att
        self.non_local_att = nn.Conv2d
        # self.dcn = torchvision.ops.deform_conv2d()
        if self.l_c == self.h_c:
            print('Channel checked!')
        else:
            raise ValueError('Low and Hih channels need to be the same!')
        # self.dcn = deform_conv2d(self.l_c,self.h_c,stride=1,padding=(0,self.r_k//2))
        self.dcn = nn.Conv2d(self.l_c, self.h_c, kernel_size=(3, 3), stride=1, padding=(self.c_k // 2, self.c_k // 2))
        self.sigmoid = nn.Sigmoid()
        if self.att == True:
            self.csa = self.non_local_att(self.l_c, self.h_c, 1, 1, 0)
        else:
            self.csa = None
        if use_process == True:
            self.preprocess = nn.Sequential(nn.Conv2d(self.l_c, self.h_c // 2, 1, 1, 0),
                                            nn.Conv2d(self.h_c // 2, self.l_c, 1, 1, 0))
        else:
            self.preprocess = None

    def forward(self, a_low, a_high):
        if self.preprocess is not None:
            a_low = self.preprocess(a_low)
            a_high = self.preprocess(a_high)
        else:
            a_low = a_low
            a_high = a_high

        a_low_c = self.dcn(a_low)
        a_low_cw = self.sigmoid(a_low_c)
        a_low_cw = a_low_cw * a_high
        a_colum = a_low + a_low_cw

        if self.csa is not None:
            a_TTOA = self.csa(a_colum)
        else:
            a_TTOA = a_colum
        return a_TTOA

#
# class SRBS_high(nn.Module):
#     def __init__(self,low_channels,high_channels,c_kernel=3,r_kernel=3):
#
#         super(SRBS_high, self).__init__()
#
#         self.l_c = low_channels
#         self.h_c = high_channels
#         self.c_k = c_kernel
#         self.r_k = r_kernel
#         self.conv = nn.Conv2d(self.l_c,self.h_c,kernel_size=(3,3),stride=1,padding=(self.c_k//2,self.c_k//2))
#         self.sigmoid = nn.Sigmoid()
#         self.conv2 = nn.Conv2d(self.l_c,self.h_c,1,1,0)
#         self.preprocess = nn.Sequential(nn.Conv2d(self.l_c,self.h_c//2,1,1,0),nn.Conv2d(self.h_c//2,self.l_c,1,1,0))
#
#     def forward(self,a_low,a_high):
#         a_low = a_low
#         a_high = self.preprocess(a_high)
#
#         a_low_c = self.conv(a_low)
#         a_low_cw = self.sigmoid(a_low_c)
#         a_low_cw = a_low_cw * a_high
#         a_colum = a_low + a_low_cw
#
#         out = self.conv2(a_colum)
#
#         return out
#
# class SRBS(nn.Module):
#     def __init__(self,low_channels,high_channels,c_kernel=3,r_kernel=3):
#
#         super(SRBS, self).__init__()
#
#         self.l_c = low_channels
#         self.h_c = high_channels
#         self.c_k = c_kernel
#         self.r_k = r_kernel
#         self.conv = nn.Conv2d(self.l_c, self.h_c, kernel_size=(3, 3), stride=1, padding=(self.c_k // 2, self.c_k // 2))
#         self.sigmoid = nn.Sigmoid()
#         self.conv2 = nn.Conv2d(self.l_c, self.h_c, 1, 1, 0)
#         self.preprocess = nn.Sequential(nn.Conv2d(self.l_c, self.h_c // 2, 1, 1, 0),
#                                         nn.Conv2d(self.h_c // 2, self.l_c, 1, 1, 0))
#     def forward(self,a_low,a_high):
#
#         a_low = self.preprocess(a_low)
#         a_high = self.preprocess(a_high)
#
#         a_low_c = self.conv(a_low)
#         a_low_cw = self.sigmoid(a_low_c)
#         a_low_cw = a_low_cw * a_high
#         a_colum = a_low + a_low_cw
#
#         out = self.conv2(a_colum)
#
#         return out
