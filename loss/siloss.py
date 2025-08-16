import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()

        self.max_depth = args.max_depth
        self.min_depth = args.min_depth

    def forward(self, pred_init, gt):
        init = pred_init

        gt = torch.clamp(gt, min=self.min_depth, max=self.max_depth)
        init = torch.clamp(init, min=self.min_depth, max=self.max_depth)


        valid_mask = gt > self.min_depth

        if self.max_depth is not None:
            valid_mask = torch.logical_and(gt > self.min_depth, gt <= self.max_depth)
        g = torch.log(init[valid_mask]) - torch.log(gt[valid_mask])
        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)

        d3 = torch.sqrt(Dg)

        return d3.mean()
