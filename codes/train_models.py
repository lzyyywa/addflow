import os
import random

from torch.utils.data.dataloader import DataLoader
import tqdm
import test as test
from loss import *
from loss import KLLoss
import torch.multiprocessing
import numpy as np
import json
import math
from torch.nn import CrossEntropyLoss  
from utils.ade_utils import emd_inference_opencv_test
from collections import Counter
from utils.hsic import hsic_normalized_cca


def cal_conditional(attr2idx, obj2idx, set_name, daset):
    def load_split(path):
        with open(path, 'r') as f:
            loaded_data = json.load(f)
        return loaded_data

    train_data = daset.train_data
    val_data = daset.val_data
    test_data = daset.test_data
    all_data = train_data + val_data + test_data
    if set_name == 'test':
        used_data = test_data
    elif set_name == 'all':
        used_data = all_data
    elif set_name == 'train':
        used_data = train_data

    v_o = torch.zeros(size=(len(attr2idx), len(obj2idx)))
    for item in used_data:
        verb_idx = attr2idx[item[1]]
        obj_idx = obj2idx[item[2]]

        v_o[verb_idx, obj_idx] += 1

    v_o_on_v = v_o / (torch.sum(v_o, dim=1, keepdim=True) + 1.0e-6)
    v_o_on_o = v_o / (torch.sum(v_o, dim=0, keepdim=True) + 1.0e-6)

    return v_o_on_v, v_o_on_o


def evaluate(model, dataset, config):
    model.eval()
    evaluator = test.Evaluator(dataset, model=None)
    all_logits, all_attr_gt, all_obj_gt, all_pair_gt, loss_avg = test.predict_logits(
        model, dataset, config)
    test_stats = test.test(
        dataset,
        evaluator,
        all_logits,
        all_attr_gt,
        all_obj_gt,
        all_pair_gt,
        config
    )
    result = ""
    key_set = ["attr_acc", "obj_acc", "ub_seen", "ub_unseen", "ub_all", "best_seen", "best_unseen", "best_hm", "AUC"]

    for key in key_set:
        result = result + key + "  " + str(round(test_stats[key], 4)) + "| "
    print(result)
    model.train()
    return loss_avg, test_stats


def save_checkpoint(state, save_path, epoch, best=False):
    filename = os.path.join(save_path, f"epoch_resume.pt")
    torch.save(state, filename)


def rand_bbox(size, lam):
    W = size[-2]
    H = size[-1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int_(W * cut_rat)
    cut_h = np.int_(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def c2c_vanilla(model, optimizer, lr_scheduler, config, train_dataset, val_dataset, test_dataset,
                scaler):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )

    model.train()
    best_loss = 1e5
    best_metric = 0
    log_training = open(os.path.join(config.save_path, 'log.txt'), 'w')

    attr2idx = train_dataset.attr2idx
    obj2idx = train_dataset.obj2idx

    train_pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                                for attr, obj in train_dataset.train_pairs]).cuda()

    config.train_pairs = train_pairs

    train_losses = []

    for i in range(config.epoch_start, config.epochs):
        progress_bar = tqdm.tqdm(
            total=len(train_dataloader), desc="epoch % 3d" % (i + 1)
        )

        epoch_train_losses = []
        epoch_cls_v_losses = []
        epoch_cls_o_losses = []
        epoch_dal_losses = []
        epoch_hem_losses = []
        epoch_com_losses = [] 
        # ==========================================
        # 【新增】：监控 Flow 和 Leak 损失的列表
        # ==========================================
        epoch_flow_losses = []
        epoch_leak_losses = []

        temp_lr = optimizer.param_groups[-1]['lr']
        print(f'Current_lr:{temp_lr}')
        for bid, batch in enumerate(train_dataloader):
            batch_img = batch[0].cuda()
            batch_verb = batch[1].cuda()
            batch_obj = batch[2].cuda()
            batch_target = batch[3].cuda()
            batch_coarse_verb = batch[4].cuda()
            batch_coarse_obj = batch[5].cuda()

            target = [batch_img, batch_verb, batch_obj, batch_target, batch_coarse_verb, batch_coarse_obj]

            # ===================== 一键分支切换：双曲 vs 欧式 =====================
            use_hyperbolic = getattr(config, 'use_hyperbolic', True)

            with torch.cuda.amp.autocast(enabled=True):
                if use_hyperbolic:
                    # 双曲分支：传入所有参数
                    predict = model(
                        video=batch_img,
                        batch_verb=batch_verb,
                        batch_obj=batch_obj,
                        batch_coarse_verb=batch_coarse_verb,
                        batch_coarse_obj=batch_coarse_obj,
                        pairs=batch_target
                    )
                    loss, loss_dict = loss_calu(predict, target, config)
                else:
                    # 欧式分支：完美兼容原生 C2C
                    predict = model(video=batch_img, pairs=batch_target)
                    
                    ce_loss_fn = CrossEntropyLoss()
                    cosine_scale = getattr(config, 'cosine_scale', 4.5) 
                    
                    # 智能解析模型输出
                    if isinstance(predict, dict):
                        verb_logits = predict['verb_logits']
                        obj_logits = predict['obj_logits']
                        pred_com_prob = predict['pred_com_prob']
                    else:
                        verb_logits, obj_logits, p_pair_v, p_pair_o = predict[0], predict[1], predict[2], predict[3]
                        pred_com_prob = p_pair_v + p_pair_o
                        
                    loss_cls_verb = ce_loss_fn(verb_logits * cosine_scale, batch_verb)
                    loss_cls_obj = ce_loss_fn(obj_logits * cosine_scale, batch_obj)
                    
                    train_v_inds = config.train_pairs[:, 0]
                    train_o_inds = config.train_pairs[:, 1]
                    pred_com_train = pred_com_prob[:, train_v_inds, train_o_inds]
                    
                    loss_com = ce_loss_fn(pred_com_train * cosine_scale, batch_target)
                    
                    loss = loss_com + 0.2 * loss_cls_verb + 0.2 * loss_cls_obj
                    
                    loss_dict = {
                        'loss_cls_verb': loss_cls_verb.item(),
                        'loss_cls_obj': loss_cls_obj.item(),
                        'loss_com': loss_com.item(),
                        'loss_dal': 0.0, 
                        'loss_hem': 0.0,
                        'loss_flow': 0.0,  # 【新增】：占位符兼容欧氏分支
                        'loss_leak': 0.0   # 【新增】：占位符兼容欧氏分支
                    }

            loss = loss / config.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if ((bid + 1) % config.gradient_accumulation_steps == 0) or (bid + 1 == len(train_dataloader)):
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_train_losses.append(loss.item() * config.gradient_accumulation_steps)
            epoch_cls_v_losses.append(loss_dict.get('loss_cls_verb', 0.0))
            epoch_cls_o_losses.append(loss_dict.get('loss_cls_obj', 0.0))
            epoch_dal_losses.append(loss_dict.get('loss_dal', 0.0))
            epoch_hem_losses.append(loss_dict.get('loss_hem', 0.0))
            epoch_com_losses.append(loss_dict.get('loss_com', 0.0))
            
            # ==========================================
            # 【新增】：安全提取 Flow 和 Leak 的值
            # ==========================================
            epoch_flow_losses.append(loss_dict.get('loss_flow', 0.0))
            epoch_leak_losses.append(loss_dict.get('loss_leak', 0.0))

            # 安全提取曲率 c 和温度 tau
            if use_hyperbolic and isinstance(predict, dict):
                current_c = predict.get('c_pos', torch.tensor(0.0)).item()
            else:
                current_c = 0.0

            if hasattr(model, 'module'):
                current_temp = F.softplus(model.module.cls_temp).item() + 0.05 if hasattr(model.module, 'cls_temp') else 0.0
            else:
                current_temp = F.softplus(model.cls_temp).item() + 0.05 if hasattr(model, 'cls_temp') else 0.0

            # ==========================================
            # 【新增】：在终端进度条显示 flow 和 leak
            # ==========================================
            progress_bar.set_postfix({
                "loss": f"{np.mean(epoch_train_losses[-50:]):.2f}",
                "v_cls": f"{np.mean(epoch_cls_v_losses[-50:]):.2f}",
                "o_cls": f"{np.mean(epoch_cls_o_losses[-50:]):.2f}",
                "com": f"{np.mean(epoch_com_losses[-50:]):.2f}",
                "dal": f"{np.mean(epoch_dal_losses[-50:]):.2f}",
                "hem": f"{np.mean(epoch_hem_losses[-50:]):.2f}",
                "flow": f"{np.mean(epoch_flow_losses[-50:]):.2f}", # <--- 让你实时看到流匹配收敛！
                "leak": f"{np.mean(epoch_leak_losses[-50:]):.2f}", # <--- 让你实时看到泄漏惩罚生效！
                "c": f"{current_c:.3f}",
                "tau": f"{current_temp:.3f}"
            })
            progress_bar.update()

        lr_scheduler.step()
        progress_bar.close()

        progress_bar.write(f"epoch {i + 1} train loss {np.mean(epoch_train_losses):.4f}")
        train_losses.append(np.mean(epoch_train_losses))

        log_training.write('\n')
        log_training.write(f"epoch {i + 1} train loss {np.mean(epoch_train_losses):.4f}\n")
        log_training.write(f"epoch {i + 1} cls_verb loss {np.mean(epoch_cls_v_losses):.4f}\n")
        log_training.write(f"epoch {i + 1} cls_obj loss {np.mean(epoch_cls_o_losses):.4f}\n")
        log_training.write(f"epoch {i + 1} com loss {np.mean(epoch_com_losses):.4f}\n")
        log_training.write(f"epoch {i + 1} dal loss {np.mean(epoch_dal_losses):.4f}\n")
        log_training.write(f"epoch {i + 1} hem loss {np.mean(epoch_hem_losses):.4f}\n")
        # ==========================================
        # 【新增】：将这两个 Loss 写入本地日志，方便以后画曲线
        # ==========================================
        log_training.write(f"epoch {i + 1} flow loss {np.mean(epoch_flow_losses):.4f}\n")
        log_training.write(f"epoch {i + 1} leak loss {np.mean(epoch_leak_losses):.4f}\n")

        if (i + 1) % config.save_every_n == 0:
            save_checkpoint({
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict(),
                'scaler': scaler.state_dict(),
            }, config.save_path, i)

        # ====== 下方的评估验证代码完全不用修改，保持你的原生逻辑 ======
        key_set = ["attr_acc", "obj_acc", "ub_seen", "ub_unseen", "ub_all", "best_seen", "best_unseen", "best_hm", "AUC"]
        if i % config.eval_every_n == 0 or i + 1 == config.epochs or i >= config.val_epochs_ts:
            print("Evaluating val dataset:")
            loss_avg, val_result = evaluate(model, val_dataset, config)
            result = ""
            for key in val_result:
                if key in key_set:
                    result = result + key + "  " + str(round(val_result[key], 4)) + "| "
            log_training.write('\n')
            log_training.write(result)
            print("Loss average on val dataset: {}".format(loss_avg))
            log_training.write('\n')
            log_training.write("Loss average on val dataset: {}\n".format(loss_avg))
            if config.best_model_metric == "best_loss":
                if loss_avg.cpu().float() < best_loss:
                    print('find best!')
                    log_training.write('find best!')
                    best_loss = loss_avg.cpu().float()
                    print("Evaluating test dataset:")
                    loss_avg, val_result = evaluate(model, test_dataset, config)
                    torch.save(model.state_dict(), os.path.join(
                        config.save_path, f"best.pt"
                    ))
                    result = ""
                    for key in val_result:
                        if key in key_set:
                            result = result + key + "  " + str(round(val_result[key], 4)) + "| "
                    log_training.write('\n')
                    log_training.write(result)
                    print("Loss average on test dataset: {}".format(loss_avg))
                    log_training.write('\n')
                    log_training.write("Loss average on test dataset: {}\n".format(loss_avg))
            else:
                if val_result[config.best_model_metric] > best_metric:
                    best_metric = val_result[config.best_model_metric]
                    log_training.write('\n')
                    print('find best!')
                    log_training.write('find best!')
                    loss_avg, val_result = evaluate(model, test_dataset, config)
                    torch.save(model.state_dict(), os.path.join(
                        config.save_path, f"best.pt"
                    ))
                    result = ""
                    for key in val_result:
                        if key in key_set:
                            result = result + key + "  " + str(round(val_result[key], 4)) + "| "
                    log_training.write('\n')
                    log_training.write(result)
                    print("Loss average on test dataset: {}".format(loss_avg))
                    log_training.write('\n')
                    log_training.write("Loss average on test dataset: {}\n".format(loss_avg))
        log_training.write('\n')
        log_training.flush()

        if i + 1 == config.epochs:
            print("Evaluating test dataset on Closed World")
            model.load_state_dict(torch.load(os.path.join(
                config.save_path, "best.pt"
            )))
            loss_avg, val_result = evaluate(model, test_dataset, config)
            result = ""
            for key in val_result:
                if key in key_set:
                    result = result + key + "  " + str(round(val_result[key], 4)) + "| "
            log_training.write('\n')
            log_training.write(result)
            print("Final Loss average on test dataset: {}".format(loss_avg))
            log_training.write('\n')
            log_training.write("Final Loss average on test dataset: {}\n".format(loss_avg))