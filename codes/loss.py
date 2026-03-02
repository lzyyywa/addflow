import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np

# 引入安全且对齐纯空间特征的底层算子
from utils.lorentz import pairwise_dist, half_aperture, oxy_angle

# ==============================================================================
# H^2EM 补充几何算子
# ==============================================================================
def dist_to_origin(x, c):
    """计算 D 维纯空间点到双曲原点 O=(1/sqrt(c), 0...) 的双曲距离"""
    # 依据流形定义逆推 time 维
    x_time = torch.sqrt(1.0 / c + torch.sum(x**2, dim=-1, keepdim=True))
    # d(O, x) = arccosh(sqrt(c) * x_time) / sqrt(c)
    return torch.acosh(torch.clamp(torch.sqrt(c) * x_time, min=1.0 + 1e-5)) / torch.sqrt(c)

# ==============================================================================
# 核心损失函数实现
# ==============================================================================

class HierarchicalEntailmentLoss(nn.Module):
    """
    层次蕴含损失 (Entailment Cone Loss)
    严格约束 child 必须包含在 parent 的蕴含锥内。
    """
    def __init__(self, K=0.1, margin=0.05):
        super().__init__()
        self.K = K
        self.margin = margin

    def forward(self, child, parent, c):
        # 1. 深度约束 (Depth Penalty): child 离原点应该比 parent 更远
        d_child = dist_to_origin(child, c)   # shape: (B, 1)
        d_parent = dist_to_origin(parent, c) # shape: (B, 1)
        loss_depth = F.relu(d_parent - d_child + self.margin)

        # 2. 锥形约束 (Cone Penalty):
        # oxy_angle 设定: 提取的是位于 parent 点的三角形外角
        theta = oxy_angle(parent, child, curv=c).unsqueeze(1)               # shape: (B, 1)
        alpha_parent = half_aperture(parent, curv=c, min_radius=self.K).unsqueeze(1) # shape: (B, 1)
        
        loss_cone = F.relu(theta - alpha_parent)

        return (loss_depth + loss_cone).mean()


class DiscriminativeAlignmentLoss(nn.Module):
    """
    判别对齐损失 (Discriminative Alignment Loss)
    通过双曲距离代替内积构建 InfoNCE 损失。
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, v_hyp, t_hyp, c):
        # 使用内置 pairwise_dist 计算 [B, B] 的距离矩阵
        dist = pairwise_dist(v_hyp, t_hyp, curv=c)
        
        # 距离转 Logits
        logits = -dist / self.temperature
        
        labels = torch.arange(v_hyp.size(0), device=v_hyp.device)
        
        loss_v2t = self.criterion(logits, labels)
        loss_t2v = self.criterion(logits.t(), labels)
        
        return (loss_v2t + loss_t2v) / 2.0


# ==============================================================================
# 总损失函数计算路由
# ==============================================================================

def loss_calu(predict, target, config):
    """
    总损失入口
    """
    batch_img, batch_verb, batch_obj, batch_pair, batch_coarse_verb, batch_coarse_obj = target
    batch_verb = batch_verb.cuda()
    batch_obj = batch_obj.cuda()
    batch_pair = batch_pair.cuda()
    
    c_pos = predict['c_pos']
    verb_logits = predict['verb_logits']
    obj_logits = predict['obj_logits']
    
    v_hyp = predict['v_hyp']                  
    o_hyp = predict['o_hyp']                  
    t_v_hyp = predict['t_v_hyp']              
    t_o_hyp = predict['t_o_hyp']              
    
    coarse_v_hyp = predict['coarse_v_hyp']    
    coarse_o_hyp = predict['coarse_o_hyp']    

    ce_loss_fn = nn.CrossEntropyLoss()
    dal_loss_fn = DiscriminativeAlignmentLoss(temperature=0.07)
    hem_loss_fn = HierarchicalEntailmentLoss(K=0.1, margin=0.05)

    # 1. 基础分类损失
    loss_cls_verb = ce_loss_fn(verb_logits, batch_verb)
    loss_cls_obj = ce_loss_fn(obj_logits, batch_obj)
    
    # 2. 判别对齐损失
    loss_dal_verb = dal_loss_fn(v_hyp, t_v_hyp, c_pos)
    loss_dal_obj = dal_loss_fn(o_hyp, t_o_hyp, c_pos)
    loss_dal = loss_dal_verb + loss_dal_obj

    # 3. 层次蕴含损失 (四大偏序链)
    loss_hem_v2fv = hem_loss_fn(child=v_hyp, parent=t_v_hyp, c=c_pos)
    loss_hem_fv2cv = hem_loss_fn(child=t_v_hyp, parent=coarse_v_hyp, c=c_pos)
    loss_hem_o2fo = hem_loss_fn(child=o_hyp, parent=t_o_hyp, c=c_pos)
    loss_hem_fo2co = hem_loss_fn(child=t_o_hyp, parent=coarse_o_hyp, c=c_pos)

    loss_hem = loss_hem_v2fv + loss_hem_fv2cv + loss_hem_o2fo + loss_hem_fo2co

    # 总损失汇总
    w_cls = getattr(config, 'w_cls', 1.0)
    w_dal = getattr(config, 'w_dal', 1.0)
    w_hem = getattr(config, 'w_hem', 1.0)

    total_loss = w_cls * (loss_cls_verb + loss_cls_obj) + \
                 w_dal * loss_dal + \
                 w_hem * loss_hem

    return total_loss

# ----------------- 保留的外部接口 -----------------
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