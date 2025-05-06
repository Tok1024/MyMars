import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.bbox import bboxDecode, iou, bbox2dist
from train.tal import TaskAlignedAssigner


class DFLoss(nn.Module):
    """Criterion class for computing DFL losses during training."""

    def __init__(self, reg_max=16) -> None:
        """Initialize the DFL module."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
                F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
                + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max=16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iouv = iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iouv) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class DetectionLoss(object):
    def __init__(self, mcfg, model):
        self.model = model
        self.mcfg = mcfg
        self.layerStrides = model.layerStrides
        self.assigner = TaskAlignedAssigner(topk=self.mcfg.talTopk, num_classes=self.mcfg.nc, alpha=0.5, beta=6.0)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.bboxLoss = BboxLoss(self.mcfg.regMax).to(self.mcfg.device)

    def preprocess(self, targets, batchSize, scaleTensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batchSize, 0, ne - 1, device=self.mcfg.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batchSize, counts.max(), ne - 1, device=self.mcfg.device)
            for j in range(batchSize):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = out[..., 1:5].mul_(scaleTensor)
        return out

    def __call__(self, preds, targets):
        """
        preds shape:
            preds[0]: (B, regMax * 4 + nc, 80, 80)
            preds[1]: (B, regMax * 4 + nc, 40, 40)
            preds[2]: (B, regMax * 4 + nc, 20, 20)
        targets shape:
            (?, 6)
        """
        loss = torch.zeros(3, device=self.mcfg.device)  # box, cls, dfl

        batchSize = preds[0].shape[0]
        no = self.mcfg.nc + self.mcfg.regMax * 4

        # predictioin preprocess
        predBoxDistribution, predClassScores = torch.cat([xi.view(batchSize, no, -1) for xi in preds], 2).split(
            (self.mcfg.regMax * 4, self.mcfg.nc), 1)
        predBoxDistribution = predBoxDistribution.permute(0, 2,
                                                          1).contiguous()  # (batchSize, 80 * 80 + 40 * 40 + 20 * 20, regMax * 4)
        predClassScores = predClassScores.permute(0, 2, 1).contiguous()  # (batchSize, 80 * 80 + 40 * 40 + 20 * 20, nc)

        # ground truth preprocess
        targets = self.preprocess(targets.to(self.mcfg.device), batchSize,
                                  scaleTensor=self.model.scaleTensor)  # (batchSize, maxCount, 5)
        gtLabels, gtBboxes = targets.split((1, 4), 2)  # cls=(batchSize, maxCount, 1), xyxy=(batchSize, maxCount, 4)
        gtMask = gtBboxes.sum(2, keepdim=True).gt_(0.0)

        # raise NotImplementedError("DetectionLoss::__call__")
        # import pdb;pdb.set_trace()
        anchor_points = self.model.anchorPoints
        stride_tensor = self.model.anchorStrides
        pred_bboxes = bboxDecode(anchor_points, predBoxDistribution, self.model.proj, xywh=False)

        # 调用TaskAlignedAssigner分配目标
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            predClassScores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gtBboxes.dtype),
            anchor_points * stride_tensor,
            gtLabels,
            gtBboxes,
            gtMask
        )

        # Cls.Loss
        target_scores_sum = max(target_scores.sum(), 1)
        loss[1] = self.bce(predClassScores, target_scores.to(predClassScores.dtype)).sum() / target_scores_sum

        # Bbox.Loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor

            loss[0], loss[2] = self.bboxLoss(
                predBoxDistribution, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum,
                fg_mask
            )

        loss[0] *= self.mcfg.lossWeights[0]  # box
        loss[1] *= self.mcfg.lossWeights[1]  # cls
        loss[2] *= self.mcfg.lossWeights[2]  # dfl

        return loss.sum()

class CWDLoss(nn.Module):
    def __init__(self, mcfg):
        self.mcfg = mcfg   
        
    def __call__(self, sfeats, tfeats):
        assert len(sfeats) == len(tfeats), "Student and teacher feature counts do not match"
        total_loss = 0.0
        count = 0
        for idx, (s_f, t_f) in enumerate(zip(sfeats, tfeats)):
            if s_f is None or t_f is None:
                print(f"[CWDLoss] Warning: s_f or t_f is None at index {idx}")
                continue
            total_loss += F.mse_loss(s_f, t_f.detach())
            count += 1
        if count == 0:
            return torch.tensor(0.0, device=self.mcfg.device)
        return total_loss / count
    
class ResponseLoss(nn.Module):
    def __init__(self, device, nc, teacherClassIndexes, temperature=2.0):
        self.device = device
        self.nc = nc #总类别数
        self.teacherClassIndexes = teacherClassIndexes
        self.T = temperature
        
    def __call__(self, sresponse, tresponse):
        total_loss = 0.0
        count = 0
        for s_out, t_out in zip(sresponse, tresponse):
            if s_out is None or t_out is None:
                continue
            # 取出类别部分
            s_cls = s_out[:, -self.nc:, :, :]
            t_cls = t_out[:, -self.nc:, :, :]
            # 只对指定类别做蒸馏
            s_cls = s_cls[:, self.teacherClassIndexes, :, :]
            t_cls = t_cls[:, self.teacherClassIndexes, :, :]
            
            B, _, H, W = s_cls.shape
            s_logits = s_cls.permute(0, 2, 3, 1).reshape(-1, self.nc)
            t_logits = t_cls.permute(0, 2, 3, 1).reshape(-1, self.nc)
            
            soft_students_log_probs = F.log_softmax(s_logits / self.T, dim=-1)
            soft_teachers_probs = (t_logits.detach() / self.T).softmax(dim=-1)
            kl_loss_pair = F.kl_div(soft_students_log_probs, soft_teachers_probs, reduction='batchmean') * (self.T * self.T)
            
            total_loss += kl_loss_pair
            count += 1
        if count == 0:
            return torch.tensor(0.0, device=self.device)
        return total_loss / count