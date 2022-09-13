import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import OrderedDict

def count_params(a):
    out = sum(p.numel() for p in a.parameters())
    return out

#Resnet
ENCODER_RESNET = [
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'resnext50_32x4d', 'resnext101_32x8d'
]
class Resnet(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True):
        super(Resnet, self).__init__()
        assert backbone in ENCODER_RESNET
        self.encoder = getattr(models, backbone)(pretrained=pretrained)
        del self.encoder.fc, self.encoder.avgpool
                
    def forward(self, x):
        features = []
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x);  features.append(x)  # 1/4
        x = self.encoder.layer2(x);  features.append(x)  # 1/8
        x = self.encoder.layer3(x);  features.append(x)  # 1/16
        x = self.encoder.layer4(x);  features.append(x)  # 1/32
        return features

    def list_blocks(self):
        lst = [m for m in self.encoder.children()]
        block0 = lst[:4]
        block1 = lst[4:5]
        block2 = lst[5:6]
        block3 = lst[6:7]
        block4 = lst[7:8]
        return block0, block1, block2, block3, block4

class CircConv(nn.Module):
    def __init__(self,in_depth,out_depth, AF='prelu',BN=False,ks=3):
        super().__init__()
        #Params
        self.in_depth = in_depth
        self.out_depth = out_depth
        self.AF = AF
        self.BN = BN
        #Layers
        self.Conv = nn.Conv2d(self.in_depth,self.out_depth,ks,1,ks//2,padding_mode='circular')
        self.prelu = nn.PReLU(self.out_depth)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(self.out_depth)
        
    def forward(self,x):
        out = self.Conv(x)
        if self.BN:
            out = self.bn(out)
        if self.AF == 'relu':
            out = self.relu(out)
        elif self.AF == 'prelu':
            out = self.prelu(out)
        elif self.AF == 'sigmoid':
            out = self.sigmoid(out)
        else:
            out = out
        return out

#Initial Convolutional block
class WConv(nn.Module):
    def __init__(self,in_depth, out_depth, AF=None, BN=False, p=0.0):
        super().__init__()
        #Params
        self.in_depth = in_depth
        self.out_depth = out_depth
        #Layers
        self.layer1 = nn.Sequential(CircConv(self.in_depth,self.in_depth//4     ,AF=AF,BN=BN,ks=1),
                                    nn.Dropout2d(p),
                                    CircConv(self.in_depth//4,self.in_depth//4  ,AF=AF,BN=BN,ks=3),
                                    nn.Dropout2d(p),
                                    CircConv(self.in_depth//4,self.out_depth    ,AF=AF,BN=BN,ks=1))

    def forward(self,x):
        out = self.layer1(x)
        return out

class Spectra(nn.Module):
    def __init__(self,in_depth,AF='prelu'):
        super().__init__()
        
        #Params
        self.in_depth = in_depth
        self.inter_depth = self.in_depth//2 if in_depth>=2 else self.in_depth

        #Layers
        self.AF1 = nn.ReLU if AF=='relu' else nn.PReLU(self.inter_depth)
        self.AF2 = nn.ReLU if AF=='relu' else nn.PReLU(self.inter_depth)
        self.inConv = nn.Sequential(nn.Conv2d(self.in_depth,self.inter_depth,1),
                                    nn.BatchNorm2d(self.inter_depth),
                                    self.AF1)
        self.midConv = nn.Sequential(nn.Conv2d(self.inter_depth,self.inter_depth,1),
                                    nn.BatchNorm2d(self.inter_depth),
                                    self.AF2)
        self.outConv = nn.Conv2d(self.inter_depth, self.in_depth, 1)
        
    def forward(self,x):
        x = self.inConv(x)
        skip = copy.copy(x)
        rfft = torch.fft.rfft2(x)
        real_rfft = torch.real(rfft)
        imag_rfft = torch.imag(rfft)
        cat_rfft = torch.cat((real_rfft,imag_rfft),dim=-1)
        cat_rfft = self.midConv(cat_rfft)
        mid = cat_rfft.shape[-1]//2
        real_rfft = cat_rfft[...,:mid]
        imag_rfft = cat_rfft[...,mid:]
        rfft = torch.complex(real_rfft,imag_rfft)
        spect = torch.fft.irfft2(rfft)
        out = self.outConv(spect + skip)
        return out

class FastFC(nn.Module):
    def __init__(self,in_depth,AF='prelu'):
        super().__init__()
        #Params
        self.in_depth = in_depth//2
        
        #Layers
        self.AF1 = nn.ReLU if AF=='relu' else nn.PReLU(self.in_depth)
        self.AF2 = nn.ReLU if AF=='relu' else nn.PReLU(self.in_depth)
        self.conv_ll = nn.Conv2d(self.in_depth,self.in_depth,3,padding='same')
        self.conv_lg = nn.Conv2d(self.in_depth,self.in_depth,3,padding='same')
        self.conv_gl = nn.Conv2d(self.in_depth,self.in_depth,3,padding='same')
        self.conv_gg = Spectra(self.in_depth, AF)
        self.bnaf1 = nn.Sequential(nn.BatchNorm2d(self.in_depth),self.AF1)
        self.bnaf2 = nn.Sequential(nn.BatchNorm2d(self.in_depth),self.AF2)

    def forward(self,x):
        mid = x.shape[1]//2
        x_loc = x[:,:mid,:,:]
        x_glo = x[:,mid:,:,:]
        x_ll = self.conv_ll(x_loc)
        x_lg = self.conv_lg(x_loc)
        x_gl = self.conv_gl(x_glo)
        x_gg = self.conv_gg(x_glo)
        out_loc = torch.add((self.bnaf1(x_ll + x_gl)),x_loc)
        out_glo = torch.add((self.bnaf2(x_gg + x_lg)),x_glo)
        out = torch.cat((out_loc,out_glo),dim=1)
        return out,out_loc,out_glo

class FourierBlock(nn.Module):
    def __init__(self,num_layer,in_depth,return_all=True):
        super().__init__()
        #Params
        self.num_layers = num_layer
        self.in_depth = in_depth
        self.return_all = return_all
        #Layers
        self.block = nn.ModuleList()
        for _ in range(self.num_layers):
            self.block.append(FastFC(self.in_depth,'prelu'))

    def forward(self,x):
        for layer in self.block:
            x,x_loc,x_glo = layer(x)
        if self.return_all:
            return x,x_loc,x_glo
        else:
            return x

class Encoder(nn.Module):
    def __init__(self,features_depth,latent_depth):
        super().__init__()
        #Params
        self.features_depth = features_depth
        self.num_maps = len(features_depth)
        self.latent_depth = latent_depth

        #Layers
        self.FB1 = FourierBlock(1,self.features_depth[0],False)
        self.down1 = nn.Upsample(scale_factor=0.5,mode='bilinear',align_corners=False)
        self.convB1 = WConv(self.features_depth[0],self.features_depth[1],AF='prelu',BN=True, p=0.4)
        self.FB2 = FourierBlock(1,self.features_depth[1],False)
        self.down2 = nn.Upsample(scale_factor=0.5,mode='bilinear',align_corners=False)
        self.convB2 = WConv(self.features_depth[1],self.features_depth[2],AF='prelu',BN=True, p=0.4)
        self.FB3 = FourierBlock(1,self.features_depth[2],False)
        self.down3 = nn.Upsample(scale_factor=0.5,mode='bilinear',align_corners=False)
        self.convB3 = WConv(self.features_depth[2],self.features_depth[3],AF='prelu',BN=True, p=0.4)
        self.FB4 = FourierBlock(1,self.features_depth[3],False)
        self.convB4 = WConv(self.features_depth[3],self.latent_depth     ,AF='prelu',BN=True, p=0.4)
        
    def forward(self,x):
        flow = x[0]
        f1 = self.FB1(flow)
        flow = torch.add(self.convB1(self.down1(f1)),x[1])
        f2 = self.FB2(flow)
        flow = torch.add(self.convB2(self.down2(f2)),x[2])
        f3 = self.FB3(flow)
        flow = torch.add(self.convB3(self.down3(f3)),x[3])
        f4 = self.FB4(flow)
        out = self.convB4(f4)
        inter_features = [f4,f3,f2,f1]
        return out,inter_features

class Decoder(nn.Module):
    def __init__(self,latent_depth,feat_depth):
        super().__init__()
        #Params
        self.in_depth = latent_depth
        self.feat_depth = feat_depth

        #Layers
        self.convB4 = WConv(self.in_depth,self.feat_depth[0],AF='prelu',BN=True)
        self.alpha4 = nn.Parameter(torch.randn((1)))
        self.FB4 = FourierBlock(1,self.feat_depth[0],False)

        self.convB3 = WConv(self.feat_depth[0],self.feat_depth[1],AF='prelu',BN=True)
        self.up3 = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)
        self.alpha3 = nn.Parameter(torch.randn((1)))
        self.FB3 = FourierBlock(1,self.feat_depth[1],False)

        self.convB2 = WConv(self.feat_depth[1],self.feat_depth[2],AF='prelu',BN=True)
        self.up2 = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)
        self.alpha2 = nn.Parameter(torch.randn((1)))
        self.FB2 = FourierBlock(1,self.feat_depth[2],False)

        self.convB1 = WConv(self.feat_depth[2],self.feat_depth[3],AF='prelu',BN=True)
        self.up1 = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)
        self.alpha1 = nn.Parameter(torch.randn((1)))
        self.FB1 = FourierBlock(1,self.feat_depth[3],False)

        self.U1 = nn.Sequential(WConv(self.feat_depth[3],self.feat_depth[4],AF='prelu',BN=True),
                                nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),
                                FourierBlock(1,self.feat_depth[4],False))

        self.U2 = nn.Sequential(WConv(self.feat_depth[4],self.feat_depth[5],AF='prelu',BN=True),
                                nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),
                                FourierBlock(1,feat_depth[5],False))
        
    def forward(self,x,inter_features):
        x = torch.add(self.convB4(x),inter_features[0]*self.alpha4[0])
        f1 = self.FB4(x)
        x = torch.add(self.up3(self.convB3(f1)),inter_features[1]*self.alpha3[0])
        f2 = self.FB3(x)
        x = torch.add(self.up2(self.convB2(f2)),inter_features[2]*self.alpha2[0])
        f3 = self.FB2(x)
        x = torch.add(self.up1(self.convB1(f3)),inter_features[3]*self.alpha1[0])
        f4 = self.FB1(x)
        f5 = self.U1(f4)
        f6 = self.U2(f5)
        upscale_features = [f1,f2,f3,f4,f5,f6]
        return upscale_features

class SemanticBranch(nn.Module):
    def __init__(self,inter_depth,num_classes,features_depth):
        super().__init__()
        #Params
        self.inter_depth = inter_depth
        self.num_classes = num_classes
        self.feat_depths = features_depth
        self.num_feat_maps = len(self.feat_depths)
        #Layers
        self.alphas = nn.ParameterList()
        self.ScaleMediator = nn.ModuleList()
        for i in range(self.num_feat_maps):
            self.alphas.append(nn.Parameter(torch.randn(1)))
            self.ScaleMediator.append(nn.Sequential(WConv(self.feat_depths[i],self.inter_depth,AF='relu',BN=True),
                                                    nn.UpsamplingBilinear2d(scale_factor=(2**(self.num_feat_maps-i-1)))))
        self.outSemanticConv = CircConv(self.inter_depth, self.num_classes, AF='relu', BN=True, ks=3)

    def forward(self,feat_list):
        out = self.ScaleMediator[0](feat_list[0])
        for i in range(1,self.num_feat_maps):
            out += self.ScaleMediator[i](feat_list[i]) * self.alphas[i]
        out = self.outSemanticConv(out)
        return out

class DepthBranch(nn.Module):
    def __init__(self,inter_depth,features_depth):
        super().__init__()
        #Params
        self.inter_depth = inter_depth
        self.feat_depths = features_depth
        self.num_feat_maps = len(self.feat_depths)
        #Layers
        self.alphas = nn.ParameterList()
        self.ScaleMediator = nn.ModuleList()
        for i in range(self.num_feat_maps):
            self.alphas.append(nn.Parameter(torch.randn(1)))
            self.ScaleMediator.append(nn.Sequential(WConv(self.feat_depths[i],self.inter_depth,AF='prelu',BN=True),
                                                    nn.UpsamplingBilinear2d(scale_factor=(2**(self.num_feat_maps-i-1)))))

        self.outDepthConv = CircConv(self.inter_depth,1,'relu',False)

    def forward(self,feat_list):
        out = self.ScaleMediator[0](feat_list[0])
        for i in range(1,self.num_feat_maps):
            out += self.ScaleMediator[i](feat_list[i]) * self.alphas[i]
        out = self.outDepthConv(out)
        return out

class FDS(nn.Module):
    ''' Main network body for semantic segmentation and depth estimation
        from single panoramas (equirectangular for now) -> Use of EquiConvs?'''
    x_mean = torch.FloatTensor(np.array([0.485,0.456,0.406])[None, :, None, None])
    x_std  = torch.FloatTensor(np.array([0.229,0.224,0.225])[None, :, None, None])

    def __init__(self,num_classes,backbone='resnet50'):
        super().__init__()
        #Params        
        ## Non-learnable parameters
        self.latent_depth = 1024 
        self.num_classes = int(num_classes)
        ## Architecture parameters
        self.backbone = backbone
        self.semantic_inter = 128
        self.depth_inter = 128
        ## Auxiliar data 
        self.features_depth = [256,512,1024,2048] #Depth from ResNet layers
        self.feat_scale = [1/8,1/4,1/2,1]
        
        #Layers
        ##Encoder part
        self.feature_extractor = Resnet(self.backbone) # resnet18, resnet50 ¿resnet101?
        self.Encoder = Encoder(self.features_depth,self.latent_depth)
        
        ##Decoder part
        self.decoder_depth = [2048,1024,512,256,64,16]
        self.Decoder = Decoder(self.latent_depth,self.decoder_depth)

        #Semantic branch
        self.depmaps_seg = [1024,512,256,64,16]
        self.SemanticSegmentator = SemanticBranch(self.semantic_inter,self.num_classes,self.depmaps_seg)

        #Depth branch
        self.depmaps_depth = [256,64,16] 
        self.DepthEstimator = DepthBranch(self.depth_inter,self.depmaps_depth)

    def forward(self,x):
        x = self._prepare_x(x) 
        output = {}

        #Feature extraction
        feature_list = self.feature_extractor(x) 
        enc_features,inter_features = self.Encoder(feature_list)

        #Decoder
        upscale_features = self.Decoder(enc_features,inter_features)
        
        #Depth prediction
        output['Depth'] = self.DepthEstimator(upscale_features[-len(self.depmaps_depth):])

        #Semantic branch
        output['Semantic'] = self.SemanticSegmentator(upscale_features[-len(self.depmaps_seg):])

        return output

    def _prepare_x(self, x):
        if self.x_mean.device != x.device:
            self.x_mean = self.x_mean.to(x.device)
            self.x_std = self.x_std.to(x.device)
        return (x[:, :3] - self.x_mean) / self.x_std

    def param_count_sections(self):
        bkb_params = count_params(self.feature_extractor)
        enc_params = count_params(self.Encoder)
        dec_params = count_params(self.Decoder)
        seg_params = count_params(self.SemanticSegmentator)
        dep_params = count_params(self.DepthEstimator)
        all_params = np.sum([bkb_params,enc_params,dec_params,seg_params,dep_params])
        print('Feature extractor parameters: %.2f M' %(bkb_params/1e6))
        print('Encoder parameters: %.2f M' %(enc_params/1e6))
        print('Decoder parameters: %.2f M' %(dec_params/1e6))
        print('Segmentation branch parameters: %.2f M' %(seg_params/1e6))
        print('Depth branch parameters: %.2f M' %(dep_params/1e6))
        print('Total number of parameters: %.2f M' %(all_params/1e6))

def save_weights(net,params,args,path):
    state_dict = OrderedDict({
                'args': args.__dict__,
                'kargs': {
                    'num_classes': params['num_classes'],
                    'backbone': params['backbone']},
                'state_dict': net.state_dict()})
    torch.save(state_dict,path)
    
def load_weigths(args):
    state_dict = torch.load(args.pth,map_location='cpu')
    net = FDS(**state_dict['kargs'])
    try:
        net.load_state_dict(state_dict['state_dict'])
    except:
        stt_dict = i3vea2local(state_dict['state_dict'])
        net.load_state_dict(stt_dict)
    return net,state_dict

def i3vea2local(state_dict):
    out_dict = OrderedDict([(k,v) for k,v in state_dict.items()])
    return out_dict

if __name__ == '__main__':
    device = torch.device('cpu')
    batch = 1
    ch,h,w = 3,512,1024
    dummy = torch.rand((batch,ch,h,w))
    resnet = FDS(14).to(device)
    output = resnet(dummy.to(device))
    resnet.save_weights(resnet,None,None)
