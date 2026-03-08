(c2c) root@interactive10085:~/C2C/38addcom-main/codes# cat loss.py
import torch.nn.functional as F
import torch.nn as nn
import torch
import math

from utils.lorentz import pairwise_dist, half_aperture, oxy_angle

class HierarchicalEntailmentLoss(nn.Module):
    def __init__(self, K=0.1):
        super().__init__()
        self.K = K

    def forward(self, child, parent, c):
        with torch.cuda.amp.autocast(enabled=False):
            theta = oxy_angle(parent.float(), child.float(), curv=c.float()).unsqueeze(1)
            alpha_parent = half_aperture(parent.float(), curv=c.float(), min_radius=self.K).unsqueeze(1)
            loss_cone = F.relu(theta - alpha_parent)
        return loss_cone.mean()


class DiscriminativeAlignmentLoss(nn.Module):
    def __init__(self, temperature=0.07, hard_weight=3.0):
        super().__init__()
        self.temperature = temperature
        self.hard_weight = hard_weight
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, v_hyp, t_hyp, c, mask_pos, mask_hard):
        with torch.cuda.amp.autocast(enabled=False):
            dist = pairwise_dist(v_hyp.float(), t_hyp.float(), curv=c.float())
            logits = -dist / self.temperature

            B = v_hyp.size(0)

            # 加上难负样本惩罚权重
            if self.hard_weight > 1.0:
                logits = logits + mask_hard.float() * math.log(self.hard_weight)

            # 【救命代码】：把 Batch 内同类的假负例彻底屏蔽成 -1e9！防止左右互搏
            false_negatives = mask_pos & ~torch.eye(B, dtype=torch.bool, device=v_hyp.device)
            logits.masked_fill_(false_negatives, -1e9)

            labels = torch.arange(B, device=v_hyp.device)

        loss_v2t = self.criterion(logits, labels)
        loss_t2v = self.criterion(logits.t(), labels)
        return (loss_v2t + loss_t2v) / 2.0


def loss_calu(predict, target, config):
    # 注意：这里解包出来的第4个参数是 batch_target (它在 train_pairs 中的索引标签)
    batch_img, batch_verb, batch_obj, batch_target, batch_coarse_verb, batch_coarse_obj = target
    batch_verb = batch_verb.cuda()
    batch_obj = batch_obj.cuda()
    batch_target = batch_target.cuda() # 确保送入 GPU，给交叉熵使用

    c_pos = predict['c_pos']
    verb_logits = predict['verb_logits']
    obj_logits = predict['obj_logits']

    # 【提取修改】：拿到未经过 softmax 污染的双曲联合 Logits
    pred_com_logits = predict['pred_com_logits']

    v_hyp = predict['v_hyp']
    o_hyp = predict['o_hyp']
    v_c_hyp = predict['v_c_hyp']
    t_v_hyp = predict['t_v_hyp']
    t_o_hyp = predict['t_o_hyp']
    t_c_hyp = predict['t_c_hyp']
    coarse_v_hyp = predict['coarse_v_hyp']
    coarse_o_hyp = predict['coarse_o_hyp']

    ce_loss_fn = nn.CrossEntropyLoss()
    dal_loss_fn = DiscriminativeAlignmentLoss(temperature=0.07, hard_weight=3.0)
    hem_loss_fn = HierarchicalEntailmentLoss(K=0.1)

    # ================= 1. C2C Vanilla 核心推断损失 (严格对齐原生) =================
    train_pairs = config.train_pairs
    train_v_inds = train_pairs[:, 0]
    train_o_inds = train_pairs[:, 1]

    # 将 [B, N_verb, N_obj] 切片为 [B, Num_Seen_Pairs]
    pred_com_train = pred_com_logits[:, train_v_inds, train_o_inds]

    # 直接用交叉熵，完美保住双曲梯度强度，对齐原生 Vanilla 逻辑！
    loss_com = ce_loss_fn(pred_com_train, batch_target)

    # ================= 2. 基元分支交叉熵 (Primitive) =================
    loss_cls_verb = ce_loss_fn(verb_logits, batch_verb)
    loss_cls_obj = ce_loss_fn(obj_logits, batch_obj)

    # ================= 3. 工具人 DAL 损失 (Discriminative Alignment) =================
    mask_verb = (batch_verb.unsqueeze(1) == batch_verb.unsqueeze(0))
    mask_obj = (batch_obj.unsqueeze(1) == batch_obj.unsqueeze(0))
    mask_pos_comp = mask_verb & mask_obj
    mask_hard_comp = mask_verb ^ mask_obj

    loss_dal = dal_loss_fn(v_c_hyp, t_c_hyp, c_pos, mask_pos=mask_pos_comp, mask_hard=mask_hard_comp)

    # ================= 4. 工具人 HEM 损失 (Hierarchical Entailment) =================
    loss_hem_vc2vs = hem_loss_fn(child=v_c_hyp, parent=v_hyp, c=c_pos)
    loss_hem_vc2vo = hem_loss_fn(child=v_c_hyp, parent=o_hyp, c=c_pos)
    loss_hem_tc2ts = hem_loss_fn(child=t_c_hyp, parent=t_v_hyp, c=c_pos)
    loss_hem_tc2to = hem_loss_fn(child=t_c_hyp, parent=t_o_hyp, c=c_pos)
    loss_hem_ts2tsp = hem_loss_fn(child=t_v_hyp, parent=coarse_v_hyp, c=c_pos)
    loss_hem_to2top = hem_loss_fn(child=t_o_hyp, parent=coarse_o_hyp, c=c_pos)

    loss_hem = loss_hem_vc2vs + loss_hem_vc2vo + \
               loss_hem_tc2ts + loss_hem_tc2to + \
               loss_hem_ts2tsp + loss_hem_to2top

    # ================= 5. 总损失融合 =================
    w_cls = getattr(config, 'w_cls', 1.0)
    w_com = getattr(config, 'w_com', 1.0)
    w_dal = getattr(config, 'w_dal', 1.0)
    w_hem = getattr(config, 'w_hem', 1.0)

    total_loss = w_cls * (loss_cls_verb + loss_cls_obj) + \
                 w_com * loss_com + \
                 w_dal * loss_dal + \
                 w_hem * loss_hem

    loss_dict = {
        'loss_cls_verb': loss_cls_verb.item(),
        'loss_cls_obj': loss_cls_obj.item(),
        'loss_com': loss_com.item(),
        'loss_dal': loss_dal.item(),
        'loss_hem': loss_hem.item()
    }

    return total_loss, loss_dict


class KLLoss(nn.Module):
    def __init__(self, error_metric=nn.KLDivLoss(size_average=True, reduce=True)):
        super().__init__()
        self.error_metric = error_metric

    def forward(self, prediction, label, mul=False):
        batch_size = prediction.shape[0]
        probs1 = F.log_softmax(prediction, 1)
        probs2 = F.softmax(label, 1)
        loss = self.error_metric(probs1, probs2)
        if mul:
            return loss * batch_size
        else:
            return loss


def hsic_loss(input1, input2, unbiased=False):
    pass


class Gml_loss(nn.Module):
    pass