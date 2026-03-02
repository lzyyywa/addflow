import torch.nn.functional as F
import torch.nn as nn
import torch
import math

from utils.lorentz import pairwise_dist, half_aperture, oxy_angle

def dist_to_origin(x, c):
    x_time = torch.sqrt(1.0 / c + torch.sum(x**2, dim=-1, keepdim=True))
    return torch.acosh(torch.clamp(torch.sqrt(c) * x_time, min=1.0 + 1e-5)) / torch.sqrt(c)

class HierarchicalEntailmentLoss(nn.Module):
    def __init__(self, K=0.1):
        super().__init__()
        self.K = K

    def forward(self, child, parent, c):
        theta = oxy_angle(parent, child, curv=c).unsqueeze(1)               
        alpha_parent = half_aperture(parent, curv=c, min_radius=self.K).unsqueeze(1) 
        loss_cone = F.relu(theta - alpha_parent)
        return loss_cone.mean()

class DiscriminativeAlignmentLoss(nn.Module):
    """
    基于 SupCon 的批次对齐损失。
    完美解决 Batch 内存在相同标签导致模型崩溃(把正确的样本推开)的问题。
    """
    def __init__(self, temperature=0.07, hard_weight=3.0):
        super().__init__()
        self.temperature = temperature
        self.hard_weight = hard_weight

    def forward(self, v_hyp, t_hyp, c, mask_pos, mask_hard_neg):
        dist = pairwise_dist(v_hyp, t_hyp, curv=c)
        logits = -dist / self.temperature
        
        if self.hard_weight > 1.0:
            logits[mask_hard_neg] += math.log(self.hard_weight)
            
        # 【核心修正】Supervised Contrastive 形式，完美包容同一 Batch 内的正样本
        # 不再使用 torch.arange 让标签相同的互相排斥！
        log_prob = F.log_softmax(logits, dim=1)
        loss_v2t = - (log_prob * mask_pos.float()).sum(dim=1) / torch.clamp(mask_pos.float().sum(dim=1), min=1.0)
        
        log_prob_t = F.log_softmax(logits.t(), dim=1)
        loss_t2v = - (log_prob_t * mask_pos.float()).sum(dim=1) / torch.clamp(mask_pos.float().sum(dim=1), min=1.0)
        
        return (loss_v2t.mean() + loss_t2v.mean()) / 2.0


def loss_calu(predict, target, config):
    batch_img, batch_verb, batch_obj, batch_pair, batch_coarse_verb, batch_coarse_obj = target
    batch_verb = batch_verb.cuda()
    batch_obj = batch_obj.cuda()
    
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
    dal_loss_fn = DiscriminativeAlignmentLoss(temperature=0.07, hard_weight=3.0)
    hem_loss_fn = HierarchicalEntailmentLoss(K=0.1)

    # 1. 基础分类损失 (不再是 9.5！)
    loss_cls_verb = ce_loss_fn(verb_logits, batch_verb)
    loss_cls_obj = ce_loss_fn(obj_logits, batch_obj)
    
    # 2. 判别对齐损失 (彻底解除了梯度崩溃互搏)
    # 计算 Batch 内动词和物品的正样本掩码
    mask_verb = (batch_verb.unsqueeze(1) == batch_verb.unsqueeze(0))
    mask_obj = (batch_obj.unsqueeze(1) == batch_obj.unsqueeze(0))
    
    # verb 的 DAL: 难负样本是 -> 动词不同，但物品相同！
    mask_pos_verb = mask_verb
    mask_hard_verb = mask_obj & (~mask_verb)
    
    # obj 的 DAL: 难负样本是 -> 物品不同，但动词相同！
    mask_pos_obj = mask_obj
    mask_hard_obj = mask_verb & (~mask_obj)

    loss_dal_verb = dal_loss_fn(v_hyp, t_v_hyp, c_pos, mask_pos_verb, mask_hard_verb)
    loss_dal_obj = dal_loss_fn(o_hyp, t_o_hyp, c_pos, mask_pos_obj, mask_hard_obj)
    loss_dal = loss_dal_verb + loss_dal_obj

    # 3. 层次蕴含损失
    loss_hem_v2fv = hem_loss_fn(child=v_hyp, parent=t_v_hyp, c=c_pos)
    loss_hem_fv2cv = hem_loss_fn(child=t_v_hyp, parent=coarse_v_hyp, c=c_pos)
    loss_hem_o2fo = hem_loss_fn(child=o_hyp, parent=t_o_hyp, c=c_pos)
    loss_hem_fo2co = hem_loss_fn(child=t_o_hyp, parent=coarse_o_hyp, c=c_pos)

    loss_hem = loss_hem_v2fv + loss_hem_fv2cv + loss_hem_o2fo + loss_hem_fo2co

    # 4. 总损失汇总
    w_cls = getattr(config, 'w_cls', 0.1)
    w_dal = getattr(config, 'w_dal', 1.0)
    w_hem = getattr(config, 'w_hem', 0.5)

    total_loss = w_cls * (loss_cls_verb + loss_cls_obj) + \
                 w_dal * loss_dal + \
                 w_hem * loss_hem

    return total_loss


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