import torch
import torch.nn as nn
import torch.nn.functional as F
from model.ours.common import *

from .Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2

class IterativeFusionModule(nn.Module):
    def __init__(self, input_dim_rgb, input_dim_depth, max_iter=3):
        super(IterativeFusionModule, self).__init__()
        midd_dim = input_dim_rgb//2
        #
        self.conv_rgbd = nn.Conv2d(input_dim_rgb + input_dim_depth, input_dim_rgb+midd_dim, kernel_size=3, padding=1)
        #
        self.conv_depth = nn.Conv2d(input_dim_depth, input_dim_rgb+midd_dim, kernel_size=3, padding=1)

        #
        self.fc_depth_out = nn.Conv2d(input_dim_rgb+midd_dim, input_dim_rgb+midd_dim, kernel_size=1)

        #
        self.max_iter = max_iter

    def forward(self, f_rgb, f_depth):

        #
        f_rgb_transformed = self.conv_rgbd(torch.cat((f_rgb, f_depth), 1))  # [b, output_dim, h, w]
        #
        f_depth_transformed = self.conv_depth(f_depth)  # [b, output_dim, h, w]

        #
        lambda_rgb, lambda_depth, _ = self.compute_lagrange_multipliers(f_rgb_transformed, f_depth_transformed)

        #
        f_fused = self.fuse_features(f_rgb_transformed, f_depth_transformed, lambda_rgb, lambda_depth)

        #
        for t in range(self.max_iter):
            #
            lambda_rgb, lambda_depth, corr = self.compute_lagrange_multipliers(f_rgb_transformed, f_fused)

            #
            f_fused = self.fuse_features(f_rgb_transformed, f_depth_transformed, lambda_rgb, lambda_depth)

        #
        predicted_depth = self.fc_depth_out(f_fused)


        return predicted_depth

    def compute_lagrange_multipliers(self, f_rgb, f_depth):
        mean_f_rgb = torch.mean(f_rgb, dim=[2, 3], keepdim=True)
        mean_f_depth = torch.mean(f_depth, dim=[2, 3], keepdim=True)

        f_rgb_centered = f_rgb - mean_f_rgb
        f_depth_centered = f_depth - mean_f_depth

        cov = torch.sum(f_rgb_centered * f_depth_centered, dim=[2, 3])

        std_f_rgb = torch.sqrt(torch.sum(f_rgb_centered ** 2, dim=[2, 3]))
        std_f_depth = torch.sqrt(torch.sum(f_depth_centered ** 2, dim=[2, 3]))

        correlation = cov / (std_f_rgb * std_f_depth)

        lambda_rgb = torch.sigmoid(correlation)
        lambda_depth = 1 - lambda_rgb

        return lambda_rgb, lambda_depth, correlation

    def fuse_features(self, f_rgb, f_depth, lambda_rgb, lambda_depth):
        f_fused = lambda_rgb[:, :, None, None] * f_rgb + lambda_depth[:, :, None, None] * f_depth
        return f_fused



class DSRN(nn.Module):
    def __init__(self, conv=default_conv):
        super(DSRN, self).__init__()

        n_feats = 64
        kernel_size = 3
        depth_ = 4
        nfeats_arr = [n_feats, n_feats+n_feats//2, 2*n_feats+n_feats//4, n_feats*3 + 24]
        nfeats_tail = n_feats*5 + 4

        # head module
        modules_head = [conv(1, nfeats_arr[0], kernel_size)]
        self.head = nn.Sequential(*modules_head)

        modules_head_rgb = [conv(3, nfeats_arr[0], kernel_size)]
        self.head_rgb = nn.Sequential(*modules_head_rgb)

        GNNblocks = []
        for i in range(depth_):
            block = IterativeFusionModule(input_dim_rgb=nfeats_arr[i], input_dim_depth=nfeats_arr[i])
            GNNblocks.append(block)
        self.gnn = nn.ModuleList(GNNblocks)

        self.c_d1 = ResidualGroup(conv, nfeats_arr[0], 3, reduction=16, n_resblocks=6)
        self.c_d2 = ResidualGroup(conv, nfeats_arr[1], 3, reduction=16, n_resblocks=4)
        self.c_d3 = ResidualGroup(conv, nfeats_arr[2], 3, reduction=16, n_resblocks=2)
        self.c_d4 = ResidualGroup(conv, nfeats_arr[3], 3, reduction=16, n_resblocks=2)

        modules_d5 = [conv(10 * n_feats+24, n_feats, 1),
                      ResidualGroup(conv, n_feats, 3, reduction=16, n_resblocks=4)]
        self.c_d5 = nn.Sequential(*modules_d5)

        self.c_r1 = conv(2 * nfeats_arr[0], nfeats_arr[0], 1)
        self.c_r2 = conv(nfeats_arr[1]+nfeats_arr[0], nfeats_arr[1], 1)
        self.c_r3 = conv(nfeats_arr[2]+nfeats_arr[0], nfeats_arr[2], 1)
        self.c_r4 = conv(nfeats_arr[3]+nfeats_arr[0], nfeats_arr[3], 1)
        self.c_r5 = conv(nfeats_tail, nfeats_arr[2], 1)

        self.c_da2_r1 = ResidualGroup(conv, nfeats_arr[0], 3, reduction=16, n_resblocks=2)
        self.c_da2_r2 = ResidualGroup(conv, nfeats_arr[0], 3, reduction=16, n_resblocks=2)
        self.c_da2_r3 = ResidualGroup(conv, nfeats_arr[0], 3, reduction=16, n_resblocks=2)
        self.c_da2_r4 = ResidualGroup(conv, nfeats_arr[0], 3, reduction=16, n_resblocks=2)

        self.conv1 = default_conv(nfeats_arr[3], 1, 3)
        self.conv2 = default_conv(nfeats_arr[2], 1, 3)

        self.sig = nn.Sigmoid()

        self.act = nn.LeakyReLU(0.1, True)

        # tail
        modules_tail = [ResidualGroup(conv, n_feats, 3, reduction=16, n_resblocks=4),
                        conv(n_feats, 1, kernel_size)]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x, dep_dav2, features_dav2):
        fda2_1 = self.c_da2_r1(features_dav2)
        fda2_2 = self.c_da2_r2(self.act(fda2_1))
        fda2_3 = self.c_da2_r3(self.act(fda2_2))
        fda2_4 = self.c_da2_r4(self.act(fda2_3))

        # head
        x = self.head(x)
        dep1 = self.c_d1(x)
        f1 = torch.cat([dep1, fda2_1], dim=1)
        dep20 = self.gnn[0](self.c_r1(f1), dep1)

        dep2 = self.c_d2(self.act(dep20))
        f2 = torch.cat([dep2, fda2_2], dim=1)
        dep30 = self.gnn[1](self.c_r2(f2), dep2)

        dep3 = self.c_d3(self.act(dep30))
        f3 = torch.cat([dep3, fda2_3], dim=1)
        dep40 = self.gnn[2](self.c_r3(f3), dep3)

        dep4 = self.c_d4(self.act(dep40))
        f4 = torch.cat([dep4, fda2_4], dim=1)
        y1 = self.c_r4(f4)
        dep50 = self.gnn[3](y1, dep4)
        dep5 = self.c_r5(self.act(dep50))

        out_rgb = self.sig(self.conv1(y1))
        out_dep = self.sig(self.conv2(dep5))

        cat1 = torch.cat([dep1, dep2, dep3, dep4, dep5], dim=1)
        dep4 = self.c_d5(cat1)
        res = dep4 + x

        # tail
        out = self.tail(res)

        return out, out_rgb, out_dep


class depthprompting(nn.Module):
    def __init__(self, args):
        super(depthprompting, self).__init__()

        self.args = args

        self.dsrn = DSRN()

        # DepthAnythingV2
        params = []
        for param in self.named_parameters():
            if param[1].requires_grad:
                params.append(param[1])

        self.depth_anything = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
        state_dict_da2 = torch.load(
            "/opt/data/private/DSR/DAv2_DSR/model/ours/Depth_Anything_V2/checkpoints/depth_anything_v2_vits.pth")
        self.depth_anything.load_state_dict(state_dict_da2)

        out_channels = [64, 64, 64, 64]
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=self.depth_anything.pretrained.embed_dim,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        self.cc1 = nn.Conv2d(
            in_channels=out_channels[0] * 4,
            out_channels=out_channels[0],
            kernel_size=1,
            stride=1,
            padding=0,
        )
        resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=2,
                stride=2,
                padding=0)
        ])
        self.resize_layer = nn.Sequential(*resize_layers)
        self.cc2 = default_conv(out_channels[0], out_channels[0], 3)
        print("Monodcular Depth Model FREEZE !!")
        print("Bias Tuning !")
        for name, var in self.depth_anything.named_parameters():
            if not 'bias' in name:
                var.requires_grad = False

    def _concat(self, fd, fe, dim=1):
        _, _, Hd, Wd = fd.shape
        _, _, He, We = fe.shape

        if Hd > He:
            h = Hd - He
            fd = fd[:, :, :-h, :]

        if Wd > We:
            w = Wd - We
            fd = fd[:, :, :, :-w]

        f = torch.cat((fd, fe), dim=dim)

        return f

    def features_extract(self, out_features, patch_h, patch_w, in_h, in_w):
        out = []
        for i, x in enumerate(out_features):
            x = x[0]
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[i](x)
            out.append(x)
        cat1 = torch.cat(out, dim=1)
        f1 = self.cc1(cat1)
        up1 = self.resize_layer(f1)
        up2 = F.interpolate(up1, size=(in_h, in_w), mode='bicubic', align_corners=True)
        up3 = self.cc2(up2)
        return up3

    def resize_to_larger_multiple_of_14(self, input_tensor):

        _, _, orig_height, orig_width = input_tensor.shape

        scale_factor = max(((orig_height + 13) // 14 * 14) / orig_height, ((orig_width + 13) // 14 * 14) / orig_width)
        new_height = int(orig_height * scale_factor)
        new_width = int(orig_width * scale_factor)
        new_height = (new_height + 13) // 14 * 14
        new_width = (new_width + 13) // 14 * 14

        resized_tensor = F.interpolate(input_tensor, size=(new_height, new_width), mode='bicubic', align_corners=False)

        return resized_tensor, (orig_height, orig_width)

    def forward(self, sample):

        rgb = sample['rgb']

        # Depth anythin v2
        image, (h, w) = self.resize_to_larger_multiple_of_14(rgb)
        patch_h, patch_w = image.shape[-2] // 14, image.shape[-1] // 14
        features = self.depth_anything.pretrained.get_intermediate_layers(image, [2, 5, 8, 11],
                                                                          return_class_token=True)
        da2_depth = self.depth_anything.depth_head(features, patch_h, patch_w)
        da2_depth = F.relu(da2_depth)

        im_features = self.features_extract(features, patch_h, patch_w, h, w)
        out = F.interpolate(da2_depth, (h, w), mode="bicubic", align_corners=True)

        dep = sample['dep']

        depth, out_rgb, out_dep = self.dsrn(dep, out, im_features)

        output = {'pred': depth, 'pred_init': out, 'out_rgb': out_rgb, 'out_dep': out_dep}

        return output
