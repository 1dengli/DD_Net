from torchvision import models as resnet_model
from models import freqsep
from models import PAF
from models import FSA
from models import resblock
from torch import nn
import torch
from thop import profile



class DD_Net(nn.Module):
    def __init__(self, channel):
        super(DD_Net, self).__init__()
        self.con1 = nn.Conv2d(channel, 3, 1)


        resnet = resnet_model.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.gf = freqsep.fre_domain(128, 16, embed_dim=512)


        self.SF = PAF.PAF(512)


        self.up6 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.conv6 = resblock.DoubleConv(512, 256)
        self.up7 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv7 = resblock.DoubleConv(256, 128)
        self.up8 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv8 = resblock.DoubleConv(128, 64)
        self.up9 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.conv10 = nn.Conv2d(32, 1, 1)

        self.b1 = FSA.FSA(256)
        self.b2 = FSA.FSA(128)
        self.b3 = FSA.FSA(64)




    def forward(self, x):

        x = self.con1(x)
        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)  # 64 64 64

        e1 = self.encoder1(e0)    # 64 64 64
        e2 = self.encoder2(e1)    # 128 32 32
        e3 = self.encoder3(e2)     # 256 16 16
        e4 = self.encoder4(e3)  # 512 8 8


        a4 = self.gf(x)
        B, N, C = a4.shape
        H = W = int(N ** 0.5)
        a4 = a4.permute(0, 2, 1).reshape(B, C, H, W)




        cat1 = self.SF(e4, a4)

        up1 = self.up6(cat1)
        e3_1 = self.b1(e3)
        cat3 = torch.cat([up1, e3_1], dim=1)

        res2 = self.conv6(cat3)
        up2 = self.up7(res2)
        e2_1 = self.b2(e2)
        cat4 = torch.cat([up2, e2_1], dim=1)

        res3 = self.conv7(cat4)
        up3 = self.up8(res3)
        e1_1 = self.b3(e1)
        cat5 = torch.cat([up3, e1_1], dim=1)

        res4 = self.conv8(cat5)
        up4 = self.up9(res4)

        out = self.conv10(up4)

        return out

if __name__ == '__main__':
    ras = DD_Net(1).cuda()
    input_tensor = torch.randn(1, 1, 128, 128).cuda()

    output = ras(input_tensor)
    print(output.shape)

    from thop import profile

    flops, params = profile(ras, (input_tensor,))
    print('Flops: %.2f G, Params: %.2f M' % (flops / 1e9, params / 1e6))
