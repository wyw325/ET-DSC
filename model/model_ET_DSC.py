import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from model.SRBS import *

class Res_CBAM_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Res_CBAM_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class ET_DSC(nn.Module):
    def __init__(self, num_classes, input_channels, block, num_blocks, nb_filter,deep_supervision=False):
        super(ET_DSC, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace = True)
        self.deep_supervision = deep_supervision
        self.pool  = nn.MaxPool2d(2, 2)
        self.up    = nn.Upsample(scale_factor=2,   mode='bilinear', align_corners=True)
        self.down  = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.avgpool = nn.AvgPool2d(3, 1, 1)
        self.up_4  = nn.Upsample(scale_factor=4,   mode='bilinear', align_corners=True)
        self.up_8  = nn.Upsample(scale_factor=8,   mode='bilinear', align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=16,  mode='bilinear', align_corners=True)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])

        # 1. 基本特征提取
        self.conv = nn.Conv2d(nb_filter[0], nb_filter[0], kernel_size=3, padding=1)

        # 2. 边缘增强（高通 + 空间卷积）
        self.spatial_conv = nn.Conv2d(
            nb_filter[0], nb_filter[0],
            kernel_size=3, padding=1, groups=nb_filter[0]  # depthwise conv
        )
        self.tau = nn.Parameter(torch.tensor(0.3))  # 初始阈值，可训练

        self.conv1_0 = self._make_layer(block, nb_filter[0],  nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1],  nb_filter[2], num_blocks[1])

        self.x2 = self._make_layer(block, nb_filter[3], nb_filter[4], num_blocks[2])  # 128-256
        self.x3 = self._make_layer(block, nb_filter[4], nb_filter[3], num_blocks[2])  # 256-128
        self.x010 = self._make_layer(block, nb_filter[0], nb_filter[1], num_blocks[2])  # 128-256
        self.x011 = self._make_layer(block, nb_filter[1], nb_filter[0], num_blocks[2])  # 256-128
        self.conv3_0 = self._make_layer(block, nb_filter[2],  nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3],  nb_filter[4], num_blocks[3])
        self.toaa01 = SRBS(nb_filter[1], nb_filter[1], c_kernel=3, r_kernel=3, use_att=True, use_process=True)
        self.toaa34 = SRBS_high(nb_filter[4], nb_filter[4], c_kernel=3, r_kernel=3, use_att=True, use_process=True)

        self.convpool = nn.Conv2d(nb_filter[0], nb_filter[1], kernel_size=3, stride=1, padding=1)
        self.convpool2 = nn.Conv2d(nb_filter[1], nb_filter[2], kernel_size=3, stride=1, padding=1)
        self.convpool3 = nn.Conv2d(nb_filter[2], nb_filter[3], kernel_size=3, stride=1, padding=1)

        self.conv0_1 = self._make_layer(block, nb_filter[0] + nb_filter[1],  nb_filter[0])
        self.conv1_1 = self._make_layer(block, nb_filter[1] + nb_filter[2],  nb_filter[1], num_blocks[0])
        self.conv2_1 = self._make_layer(block, nb_filter[2] + nb_filter[3],  nb_filter[2], num_blocks[1])
        self.conv3_1 = self._make_layer(block, nb_filter[3] + nb_filter[4],  nb_filter[3], num_blocks[2])

        self.conv0_2 = self._make_layer(block, nb_filter[0]*2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = self._make_layer(block, nb_filter[1]*2 + nb_filter[2], nb_filter[1], num_blocks[0])
        self.conv2_2 = self._make_layer(block, nb_filter[2]*2 + nb_filter[3], nb_filter[2], num_blocks[1])

        self.conv0_3 = self._make_layer(block, nb_filter[0]*3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = self._make_layer(block, nb_filter[1]*3 + nb_filter[2], nb_filter[1], num_blocks[0])

        self.conv0_4 = self._make_layer(block, nb_filter[0]*4 + nb_filter[1], nb_filter[0])

        self.conv0_4_final = self._make_layer(block, nb_filter[0]*5, nb_filter[0])

        self.conv0_4_1x1 = nn.Conv2d(nb_filter[4], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_3_1x1 = nn.Conv2d(nb_filter[3], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_2_1x1 = nn.Conv2d(nb_filter[2], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_1_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)

        if self.deep_supervision:
            self.final1 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final  = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)

    def _make_layer(self, block, input_channels,  output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = self.conv0_0(input)

        xedge = x0_0 - self.avgpool(x0_0)  # 高通成分
        xedge = self.spatial_conv(xedge)   # 空间卷积学习显著性
        xedge = torch.sigmoid(xedge)
        # xedge = F.relu(xedge - self.tau)
        x0_0 = x0_0 * xedge

        # # Step 4: 语义线 (浅层边缘的语义embedding)
        # semantic_line = self.semantic_proj(x0_0)  # (B, semantic_dim, 1, 1)
        semantic_line = self.pool(xedge)
        semantic_line = self.convpool(semantic_line)
        semantic_line = self.pool(semantic_line)
        semantic_line = self.convpool2(semantic_line)
        semantic_line = self.pool(semantic_line)
        semantic_line = self.convpool3(semantic_line)


        x1_0 = self.conv1_0(self.pool(x0_0))

        xx = self.x010(x0_0)
        xsrbs = self.toaa01(xx, self.up(x1_0))
        x0_1 = self.x011(xsrbs)

        x2_0 = self.conv2_0(self.pool(x1_0))
        xsup = self.final1(x0_1)
        xsup = self.pool(xsup)
        xsup = self.pool(xsup)
        xsup = self.pool(xsup)
        xsup = torch.sigmoid(xsup)


        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x3_0 = x3_0 * semantic_line



        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        a = (1 - xsup) * x3_0
        b = xsup * self.up(x4_0)

        x2 = self.x2(a)
        xsrbs = self.toaa34(x2, b)
        x3_1 = self.x3(xsrbs)
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        Final_x0_4 = self.conv0_4_final(
            torch.cat([self.up_16(self.conv0_4_1x1(x4_0)),self.up_8(self.conv0_3_1x1(x3_1)),
                       self.up_4 (self.conv0_2_1x1(x2_2)),self.up  (self.conv0_1_1x1(x1_3)), x0_4], 1))
        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(Final_x0_4)
            return [output1, output2, output3, output4]
        else:
            output = self.final4(Final_x0_4)
            return output

def main():
    # 模拟类别数量，可根据实际任务调整
    num_classes = 1
    # 输入图像的通道数
    input_channels = 3
    # 不同阶段的滤波器数量，示例设置，可按需修改
    nb_filter = [16, 32, 64, 128, 256]
    # 每个阶段的模块重复数量，示例设置，可按需修改
    num_blocks = [2, 2, 2, 2]
    # 是否开启深度监督
    deep_supervision = True
    # 创建DNANet模型实例
    model = ET_DSC(num_classes, input_channels, Res_CBAM_block, num_blocks, nb_filter, deep_supervision)

    image_tensor = torch.randn(1, 3, 256, 256)
    model.eval()
    output = model(image_tensor)
    if deep_supervision:
        for i, out in enumerate(output):
            print(f"Output {i + 1} shape: {out.shape}")
    else:
        print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    main()