import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init



class patch_wise_attention_layer(nn.Module):
    def __init__(self, in_channel, kernel_size):
        super(patch_wise_attention_layer, self).__init__()
        # in_channel 은 kernel_size의 배수여야 한다
        self.in_channel = in_channel #
        self.kernel_size = kernel_size
        self.stride = self.kernel_size
        self.padding = 0

        self.patch_block = nn.Conv2d(self.in_channel, self.in_channel, kernel_size = self.kernel_size, stride = self.stride)
        self.unfold = nn.Unfold(3,1,0,1)
        self.fp = nn.ReflectionPad2d(1)

    def forward(self, x):
        # bs , c, h, w 
        # print(x)
        in_dim = x.shape
        bs, ch, h, w = in_dim
        patch_cnt = h // self.kernel_size
        x1 = self.patch_block(x) # [bs, c, kernel_size, kernel_size]
        x = x1.view([bs*ch, -1, patch_cnt, patch_cnt]) # 채널 별 패치 간 연산을 위해
        x = self.fp(x)
        x = self.unfold(x)
        x = x.view([bs, ch, 3*3, -1]).permute((0,3,1,2)).contiguous().view([-1, ch, 3*3])
        xT = x.transpose(1,2)[:,0,:].view(-1,1,ch) # 연산량 kernel_size 배 감소
        
        xxT = torch.bmm(xT,x) # 확인

        xxT = xxT.view([bs* patch_cnt *patch_cnt, -1])
        xxT = torch.mean(xxT, dim = 1).view([bs, 1, patch_cnt, patch_cnt]) # sum or mean 
        y = xxT.repeat(1, ch, 1, 1)
                
        out = x1*y
        # out dimension = [bs, c, h/k, w/h] downsampling 
        return out


class GaussianDiffusionTrainer(nn.Module):
    # beta_1 ~ beta_T 구간을 T개의 step으로 동일한 간격으로 나누고
    def __init__(self, scale_factor):
        super().__init__()
        self.sf = scale_factor
        
    def forward(self, x_0):
        noise = torch.randn_like(x_0)
        x_t = (1. - self.sf)* x_0 + self.sf* noise
        
        return x_t


class G_ELAN(nn.Module):
    def __init__(self, in_channel, phase):
        super(G_ELAN, self).__init__()
        self.in_channel = in_channel
        self.phase = phase
        self.cc_block = nn.Conv2d(self.in_channel, self.in_channel *2 , 1)
        self.GN1 = GaussianDiffusionTrainer(0.10)
        self.GN2 = GaussianDiffusionTrainer(0.15)
        self.GN3 = GaussianDiffusionTrainer(0.20)
        self.GN4 = GaussianDiffusionTrainer(0.25)
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels = self.in_channel, out_channels = self.in_channel // 2, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.in_channel, momentum = 0.1),
            nn.GELU(),
        )
        
        self.conv1x1 = nn.Conv2d(self.in_channel*4 , self.in_channel, kernel_size = 1, stride = 1 , padding = 0)
        

    def forward(self, x):
        
        # c-> 2c
        x = self.cc_block(x)

        # partial
        l = x[:,:self.in_channel,:,:] 
        r = x[:,self.in_channel:,:,:]

        b = self.conv2d(r)
        if self.phase == "train":
            b = self.GN1(b)
        b = torch.add(b, r)

        b1 = self.conv2d(b)
        if self.phase == "train":
            b1 = self.GN2(b1)
        b1 = torch.add(b, b1)

        b2 = self.conv2d(b1)
        if self.phase == "train":
            b2 = self.GN3(b2)
        b2 = torch.add(b1, b2)

        b3 = self.conv2d(b2)
        if self.phase == "train":
            b3 = self.GN4(b3)
        b3 = torch.add(b2, b3)

        if self.phase == "train":
            l = self.GN1(l)
            r = self.GN1(r)

        c = torch.cat([l, r, b1, b3 ], dim = 1)
        out = self.conv1x1(c) 

        return out
        

class front_layer(nn.Module):
    def __init__(self):
        super(front_layer, self).__init__()

        self.input_block = nn.Conv2d(3, 32, 3, 1, 1)
        self.mp = nn.MaxPool2d(2, 2)
        
        self.first_layer = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64, momentum = 0.1),
            nn.GELU()
        )
        self.mp1 = nn.Conv2d(64, 64, 3, 2, 1 )

        self.second_layer = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128, momentum = 0.1),
            nn.GELU()
        )

        self.mp2 = nn.Conv2d(128, 128, 3, 2, 1 )

        self.last_layer = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256, momentum = 0.1),
            nn.GELU()
        )
        
    def forward(self, x):
        
        # input = image to tensor
        # input dimension = [bs, 3, 416, 416]

        x = self.input_block(x)
        x = self.first_layer(x)
        x= self.mp1(x)
        x = self.second_layer(x)
        x = self.mp2(x)

        # output = feature map tensor
        # output dimension = [bs 256 104 104]

        x = self.last_layer(x)
        
        return x

class rear_layer(nn.Module):
    def __init__(self, in_channel = 256, num_cls =80 , phase = 'train'):
        super(rear_layer, self).__init__()
        #
        self.anchors = np.array([[10,13],  [16,30],  [33,23],  [30,61],  [62,45],  [59,119],  [116,90],  [156,198],  [373,326]], dtype = np.float64)
        self.mask = [[0,1,2],[3,4,5],[6,7,8]]
        self.phase = phase
        self.num_cls = num_cls

        self.c1 = in_channel
        self.attention1 = patch_wise_attention_layer(self.c1, 2)
        self.GELAN1 = G_ELAN(self.c1, self.phase)
        self.c2 = self.c1 * 2
        self.conv1x1_1 = nn.Conv2d(self.c1, self.c2, 1, stride = 1, padding = 0)
        self.attention2= patch_wise_attention_layer(self.c2, 2)
        self.GELAN2 = G_ELAN(self.c2, self.phase)

        self.c3 = self.c2 * 2
        self.conv1x1_2 = nn.Conv2d(self.c2, self.c3, 1, stride = 1, padding = 0)
        self.attention3= patch_wise_attention_layer(self.c3, 2)
        self.GELAN3 = G_ELAN(self.c3 , self.phase)

        self.conv3_1 = nn.Conv2d(self.c3, 512, 3, 1, 1)
        self.conv2_1 = nn.Conv2d(self.c2, 256, 3, 1, 1)
        self.conv1_1 = nn.Conv2d(self.c1, 128, 3, 1, 1)

        self.conv3 = nn.Conv2d(self.c3 // 2, (self.num_cls + 5) *3 , kernel_size = 1, stride = 1, padding = 0)        
        self.conv2 = nn.Conv2d(self.c2 // 2, (self.num_cls + 5) *3 , kernel_size = 1, stride = 1, padding = 0)
        self.conv1 = nn.Conv2d(self.c1 // 2, (self.num_cls + 5) *3 , kernel_size = 1, stride = 1, padding = 0)

        # [ bs x in_channel *4 x 13 x 13 ] 
        self.detect3 = detect_layer(self.anchors[self.mask[2]], self.num_cls, [416, 416],  self.phase, 'default')
        self.detect2 = detect_layer(self.anchors[self.mask[1]], self.num_cls, [416, 416],  self.phase, 'default')
        self.detect1 = detect_layer(self.anchors[self.mask[0]], self.num_cls, [416, 416],  self.phase, 'default')
        
        self.aux2 = nn.Conv2d(self.c2 + self.c1, self.c1, 3, 1, 1)
        self.aux1 = nn.Conv2d(self.c1 + self.c1//2 , self.c1//2 ,3,1,1)
        self.upsample = nn.Upsample(scale_factor = 2, mode = "nearest")

    def forward(self, x):
         
        x1 = self.attention1(x)
        x1 = self.GELAN1(x1) 

        x2 = self.conv1x1_1(x1) # 채널 2배
        x2 = self.attention2(x2)
        x2 = self.GELAN2(x2)

        x3 = self.conv1x1_2(x2)
        x3 = self.attention3(x3)
        x3 = self.GELAN3(x3) # bs x 1024 x 13 x 13

        x1_1 = self.conv1_1(x1)
        x2_1 = self.conv2_1(x2)
        x3_1 = self.conv3_1(x3)

        aux3 = x3_1
        aux2 = self.upsample(x3_1)
        aux2 = torch.cat([aux2, x2_1], dim = 1) # 738 x 26 x 26
        aux2 = self.aux2(aux2) # 256 26 26

        aux1 = self.upsample(aux2) # 256 52 52
        aux1 = torch.cat([aux1, x1_1], dim = 1) # 384 52 52
        aux1 = self.aux1(aux1)
        
        out1 = self.conv1(aux1)
        out2 = self.conv2(aux2)
        out3 = self.conv3(aux3)

        out3 = self.detect3(out3)
        out2 = self.detect2(out2)
        out1 = self.detect1(out1)

        return [out3, out2, out1]

class ODNetwork(nn.Module):
    def __init__(self, phase, num_cls):
        super(ODNetwork, self).__init__()
        self.num_cls = num_cls
        self.phase = phase

        self.front_network = front_layer()
        self.rear_network = rear_layer(in_channel = 256, num_cls = self.num_cls, phase = self.phase) #

    def forward(self, x):

        x = self.front_network(x)
        print(x.shape)
        out = self.rear_network(x)
        print(out[0][0].shape)
        return out

class detect_layer(nn.Module):
    def __init__(self, anchors, nc, img_size, phase, arc ):
        super(detect_layer,self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.na = len(self.anchors)
        self.nc = nc
        self.no = nc + 5
        self.nx = 0
        self.ny = 0
        self.img_size= img_size
        self.phase = phase
        self.arc = arc
        self.oi = [0, 1, 2, 3] + list(range(5, self.no))  # output indices

    def forward(self, p):
        bs, _, ny, nx = p.shape
        if (self.nx, self.ny) != (nx, ny):
            create_grids(self, self.img_size, (nx, ny), p.device, p.dtype)
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.phase == "train":
            return p
        if self.phase == "inference":
            io = p.clone()  # inference output
            io[..., 0:2] = torch.sigmoid(io[..., 0:2]) + self.grid_xy  # xy
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
            # io[..., 2:4] = ((torch.sigmoid(io[..., 2:4]) * 2) ** 3) * self.anchor_wh  # wh power method
            io[..., :4] *= self.stride

            torch.sigmoid_(io[..., 4:])
 
            # compute conf
            io[..., 5:] *= io[..., 4:5]  # conf = obj_conf * cls_conf
            # output :
            # torch.Size([1, 312, 8])   8x13
            # torch.Size([1, 1248, 8])   16x26
            # torch.Size([1, 4992, 8]    32x52
            return io[..., self.oi].view(bs, -1, self.no - 1), p


def create_grids(self, img_size=416, ng=(13, 13), device='cpu', type=torch.float32):
    nx, ny = ng  # x and y grid size
    self.img_size = max(img_size)
    self.stride = self.img_size / max(ng)

    # build xy offsets
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    self.grid_xy = torch.stack((xv, yv), 2).to(device).type(type).view((1, 1, ny, nx, 2))

    # build wh gains
    self.anchor_vec = self.anchors.to(device) / self.stride
    self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2).to(device).type(type)
    self.ng = torch.Tensor(ng).to(device)
    self.nx = nx
    self.ny = ny