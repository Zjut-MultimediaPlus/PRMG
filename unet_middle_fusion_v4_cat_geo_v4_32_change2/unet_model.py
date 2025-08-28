""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, geo_channels, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # IR
        incha = 32
        self.IR_inc = (DoubleConv(8, incha))
        self.IR_down1 = (Down(incha, incha*2))
        self.IR_down2 = (Down(incha*2, incha*4))
        self.IR_down3 = (Down(incha*4, incha*8))

        # PMW
        incha = 32
        self.PMW_inc = (DoubleConv(13, incha))
        self.PMW_down1 = (Down(incha, incha*2))
        self.PMW_down2 = (Down(incha*2, incha*4))
        self.PMW_down3 = (Down(incha*4, incha*8))

        # PR
        incha = 32
        self.PR_inc = (DoubleConv(176, incha))
        self.PR_down1 = (Down(incha, incha*2))
        self.PR_down2 = (Down(incha*2, incha*4))
        self.PR_down3 = (Down(incha*4, incha*8))

        # self.down1 = (Down(64, 128))
        # self.down2 = (Down(128, 256))
        # self.down3 = (Down(256, 512))
        incha = 32
        factor = 2 if bilinear else 1
        self.down4 = (Down(incha*24, incha*48 // factor))
        self.up1 = (Up(incha*48, incha*24 // factor, bilinear))
        self.up2 = (Up(incha*24, incha*12 // factor, bilinear))
        self.up3 = (Up(incha*12, incha*6 // factor, bilinear))
        self.up4 = (Up(incha*6, incha*3, bilinear))

        self.geo_inc = (DoubleConv(geo_channels, incha))
        self.x_geo_inc = (DoubleConv(incha*4, incha*2))
        self.x_geo_inc1 = (DoubleConv(incha*2, incha))
        self.x_geo_inc2 = (DoubleConv(incha, 16))

        self.outc = (OutConv(16, n_classes))




    def forward(self, x,geo):
        # IR
        IR_x1 = self.IR_inc(x[:, :8])
        IR_x2 = self.IR_down1(IR_x1)
        IR_x3 = self.IR_down2(IR_x2)
        IR_x4 = self.IR_down3(IR_x3)

        # PMW
        PMW_x1 = self.PMW_inc(x[:, 8:21])
        PMW_x2 = self.PMW_down1(PMW_x1)
        PMW_x3 = self.PMW_down2(PMW_x2)
        PMW_x4 = self.PMW_down3(PMW_x3)

        # PR
        PR_x1 = self.PR_inc(x[:, 21:])
        PR_x2 = self.PR_down1(PR_x1)
        PR_x3 = self.PR_down2(PR_x2)
        PR_x4 = self.PR_down3(PR_x3)

        # cat
        x_cat1 = torch.cat((IR_x1, PMW_x1, PR_x1), dim=1)
        x_cat2 = torch.cat((IR_x2, PMW_x2, PR_x2), dim=1)
        x_cat3 = torch.cat((IR_x3, PMW_x3, PR_x3), dim=1)
        x_cat4 = torch.cat((IR_x4, PMW_x4, PR_x4), dim=1)

        # up
        x5 = self.down4(x_cat4)
        x = self.up1(x5, x_cat4)
        x = self.up2(x, x_cat3)
        x = self.up3(x, x_cat2)
        x = self.up4(x, x_cat1)

        geo1 = self.geo_inc(geo)
        x_geo_cat = torch.cat((x, geo1), dim=1)
        x_geo_cat = self.x_geo_inc(x_geo_cat)
        x_geo_cat = self.x_geo_inc1(x_geo_cat)
        x_geo_cat = self.x_geo_inc2(x_geo_cat)
        logits = self.outc(x_geo_cat)

        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)