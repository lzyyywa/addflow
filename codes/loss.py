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

            false_negatives = mask_pos & ~torch.eye(B, dtype=torch.bool, device=v_hyp.device)
            logits.masked_fill_(false_negatives, -1e9)

            labels = torch.arange(B, device=v_hyp.device)

        loss_v2t = self.criterion(logits, labels)
        loss_t2v = self.criterion(logits.t(), labels)
        return (loss_v2t + loss_t2v) / 2.0


def loss_calu(predict, target, config):

    batch_img, batch_verb, batch_obj, batch_target, batch_coarse_verb, batch_coarse_obj = target
    batch_verb = batch_verb.cuda()
    batch_obj = batch_obj.cuda()
    batch_target = batch_target.cuda() 

    c_pos = predict['c_pos']
    verb_logits = predict['verb_logits']
    obj_logits = predict['obj_logits']

    pred_com_logits = predict['pred_com_logits']

    v_hyp = predict['v_hyp']
    o_hyp = predict['o_hyp']
    v_c_hyp = predict['v_c_hyp']
    t_v_hyp = predict['t_v_hyp']
    t_o_hyp = predict['t_o_hyp']
    t_c_hyp = predict['t_c_hyp']
    coarse_v_hyp = predict['coarse_v_hyp']
    coarse_o_hyp = predict['coarse_o_hyp']

    # =====================================================================
    # 【新增】：安全提取 Flow 和 Leakage 预测值 (支持退化回 Vanilla 模式)
    # =====================================================================
    flow_pred_v = predict.get('flow_pred_v', None)
    flow_target_v = predict.get('flow_target_v', None)
    flow_pred_o = predict.get('flow_pred_o', None)
    flow_target_o = predict.get('flow_target_o', None)
    leak_v_logits = predict.get('leak_v_logits', None)
    leak_o_logits = predict.get('leak_o_logits', None)

    ce_loss_fn = nn.CrossEntropyLoss()
    dal_loss_fn = DiscriminativeAlignmentLoss(temperature=0.07, hard_weight=3.0)
    hem_loss_fn = HierarchicalEntailmentLoss(K=0.1)

    train_pairs = config.train_pairs
    train_v_inds = train_pairs[:, 0]
    train_o_inds = train_pairs[:, 1]

    pred_com_train = pred_com_logits[:, train_v_inds, train_o_inds]

    loss_com = ce_loss_fn(pred_com_train, batch_target)

    # 1. 分支交叉熵 (原生 C2C 损失)
    loss_cls_verb = ce_loss_fn(verb_logits, batch_verb)
    loss_cls_obj = ce_loss_fn(obj_logits, batch_obj)

    # 2. DAL 损失 (双曲判别对齐，严格文献公式)
    mask_verb = (batch_verb.unsqueeze(1) == batch_verb.unsqueeze(0))
    mask_obj = (batch_obj.unsqueeze(1) == batch_obj.unsqueeze(0))
    mask_pos_comp = mask_verb & mask_obj
    mask_hard_comp = mask_verb ^ mask_obj

    loss_dal = dal_loss_fn(v_c_hyp, t_c_hyp, c_pos, mask_pos=mask_pos_comp, mask_hard=mask_hard_comp)

    # 3. HEM 损失 (双曲层级蕴含，严格文献公式)
    loss_hem_vc2vs = hem_loss_fn(child=v_c_hyp, parent=v_hyp, c=c_pos)
    loss_hem_vc2vo = hem_loss_fn(child=v_c_hyp, parent=o_hyp, c=c_pos)
    loss_hem_tc2ts = hem_loss_fn(child=t_c_hyp, parent=t_v_hyp, c=c_pos)
    loss_hem_tc2to = hem_loss_fn(child=t_c_hyp, parent=t_o_hyp, c=c_pos)
    loss_hem_ts2tsp = hem_loss_fn(child=t_v_hyp, parent=coarse_v_hyp, c=c_pos)
    loss_hem_to2top = hem_loss_fn(child=t_o_hyp, parent=coarse_o_hyp, c=c_pos)

    loss_hem = loss_hem_vc2vs + loss_hem_vc2vo + \
               loss_hem_tc2ts + loss_hem_tc2to + \
               loss_hem_ts2tsp + loss_hem_to2top

    # =====================================================================
    # 4. 【新增】：Flow Matching 流动匹配损失 (MSE)
    # 文献对齐：FlowComposer Eq.(4) L = || v(x_t) - (x_1 - x_0) ||^2
    # 这里 target 已经是 (Text - Visual)，所以直接算 MSE 即可保证欧氏轨迹。
    # =====================================================================
    loss_flow = torch.tensor(0.0, device=batch_verb.device)
    if flow_pred_v is not None and flow_target_v is not None:
        loss_flow = loss_flow + F.mse_loss(flow_pred_v, flow_target_v)
    if flow_pred_o is not None and flow_target_o is not None:
        loss_flow = loss_flow + F.mse_loss(flow_pred_o, flow_target_o)

    # =====================================================================
    # 5. 【新增】：Leakage-Guided 泄漏引导交叉损失 (CE)
    # 文献对齐：FlowComposer Section 4.3 
    # 逻辑验证：强制残留着动词信息的物体视觉特征(leak_v_logits)预测出正确的动词标签；反之亦然。
    # =====================================================================
    loss_leak = torch.tensor(0.0, device=batch_verb.device)
    if leak_v_logits is not None and leak_o_logits is not None:
        loss_leak_v = ce_loss_fn(leak_v_logits, batch_verb)
        loss_leak_o = ce_loss_fn(leak_o_logits, batch_obj)
        loss_leak = loss_leak_v + loss_leak_o


    # =====================================================================
    # 总损失多目标联合优化 (Multi-Objective Optimization)
    # =====================================================================
    w_cls = getattr(config, 'w_cls', 1.0)
    w_com = getattr(config, 'w_com', 1.0)
    w_dal = getattr(config, 'w_dal', 1.0)
    # HEM权重极其关键！双曲结构极易压制分类特征，需保持 0.1 左右
    w_hem = getattr(config, 'w_hem', 0.1) 
    
    # 提取我们在 yaml 中新加的权重，默认值遵循顶会标准
    w_flow = getattr(config, 'w_flow', 1.0)
    w_leak = getattr(config, 'w_leak', 0.5)

    total_loss = w_cls * (loss_cls_verb + loss_cls_obj) + \
                 w_com * loss_com + \
                 w_dal * loss_dal + \
                 w_hem * loss_hem + \
                 w_flow * loss_flow + \
                 w_leak * loss_leak

    loss_dict = {
        'loss_cls_verb': loss_cls_verb.item(),
        'loss_cls_obj': loss_cls_obj.item(),
        'loss_com': loss_com.item(),
        'loss_dal': loss_dal.item(),
        'loss_hem': loss_hem.item(),
        # 兼容纯 C2C Vanilla 模式下没有新 loss 的情况
        'loss_flow': loss_flow.item() if isinstance(loss_flow, torch.Tensor) else 0.0,
        'loss_leak': loss_leak.item() if isinstance(loss_leak, torch.Tensor) else 0.0
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