import math
import torch
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import scipy.io as sio
from interpretation import process_pattern
from tqdm import tqdm
import utils

def eval_one_epoch(hint, model, dataset, val_flag='val', interpretation=False, time_prediction=False):
    val_acc, val_ap, val_f1, val_auc = [], [], [], []
    y_true, y_pred, y_score, y_one_hot_np = None, None, None, None
    dataset.reset()
    model.test = True
    device = model.n_feat_th.data.device
    if interpretation:
        roc_auc_score = utils.roc_auc_score_single
    else:
        roc_auc_score = utils.roc_auc_score_multi
    if val_flag == 'train':
        num_test_instance = dataset.get_size()
        get_sample = dataset.train_samples
        dataset.initialize()
    elif val_flag == 'val':
        num_test_instance = dataset.get_val_size()
        get_sample = dataset.val_samples
        dataset.initialize_val()
    elif val_flag == 'test':
        num_test_instance = dataset.get_test_size()
        get_sample = dataset.test_samples
        dataset.initialize_test()
    with torch.no_grad():
        model = model.eval()
        TEST_BATCH_SIZE = dataset.bs
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
        num_class = dataset.num_class
        walk_pattern = None
        walk_score = None
        walk_pattern_total = None
        walk_pattern_label_total = None
        walk_score_total = None
        NLL_total = None
        time_predicted_total = None
        time_gt_total = None
        loop_num = num_test_batch
        for k in tqdm(range(loop_num)):
            src_1_l_cut, src_2_l_cut, dst_l_cut, ts_l_cut, e_l_cut, true_label = get_sample()
            if (interpretation):
                pred_score, pattern_score = model.contrast(src_1_l_cut, src_2_l_cut, dst_l_cut, ts_l_cut, e_l_cut)
                # interpretation
                src_1_walks_score, src_2_walks_score, tgt_walks_score, src_1_walk_pattern, src_2_walk_pattern, tgt_walk_pattern = pattern_score
                # deal with scores and pattern
                # reshape, cpu, numpy scores
                """Note Softmax should not be applied here"""
                src_1_walks_score = src_1_walks_score.detach().cpu().numpy()
                src_2_walks_score = src_2_walks_score.detach().cpu().numpy()
                tgt_walks_score = tgt_walks_score.detach().cpu().numpy()
                walk_pattern_label = true_label.repeat(src_1_walk_pattern.shape[-1]).reshape(1,-1).repeat(3,0).reshape(-1) # each node has src_1_walk_pattern.shape[-1] walks(128), totally 3 nodes
                walk_pattern = np.concatenate([src_1_walk_pattern.reshape(-1), src_2_walk_pattern.reshape(-1),tgt_walk_pattern.reshape(-1)])
                assert(len(walk_pattern_label == 0) == len(walk_pattern_label == 1))
                
                if walk_pattern_total is None:
                    walk_pattern_total = walk_pattern
                    walk_pattern_label_total = walk_pattern_label
                else:
                    walk_pattern_total = np.concatenate([walk_pattern_total, walk_pattern])
                    walk_pattern_label_total = np.concatenate([walk_pattern_label_total, walk_pattern_label])
                walk_score = np.concatenate([src_1_walks_score, src_2_walks_score, tgt_walks_score])
                if walk_score_total is None:
                    walk_score_total = walk_score
                else:
                    walk_score_total = np.concatenate([walk_score_total, walk_score])
            elif time_prediction:
                true_label_torch = torch.from_numpy(true_label).to(device)
                _NLL_score, _ = model.contrast(src_1_l_cut, src_2_l_cut, dst_l_cut, ts_l_cut, e_l_cut, endtime_pos=true_label_torch)
                ave_mae_t, ave_log_t, NLL_score, time_list = _NLL_score
                # we compare the log t distribution
                time_predicted = time_list[0].detach().cpu().numpy()
                time_gt = time_list[1].detach().cpu().numpy()
                if time_predicted_total is None:
                    time_predicted_total = time_predicted
                    time_gt_total = time_gt
                else:
                    time_predicted_total = np.concatenate([time_predicted_total, time_predicted])
                    time_gt_total = np.concatenate([time_gt_total, time_gt])

            else:
                pred_score, _ = model.contrast(src_1_l_cut, src_2_l_cut, dst_l_cut, ts_l_cut, e_l_cut)
            if time_prediction:
                if NLL_total is None:
                    NLL_total = NLL_score
                    MSE_total = ave_log_t
                    MAE_total = ave_mae_t
                else:
                    NLL_total += NLL_score
                    MSE_total += ave_log_t
                    MAE_total += ave_mae_t
            else:
                pred_label = torch.argmax(pred_score, dim=1).cpu().detach().numpy()
                pred_score = torch.nn.functional.softmax(pred_score, dim=1).cpu().numpy()
                y_one_hot = torch.nn.functional.one_hot(torch.from_numpy(true_label).long(), num_classes=model.num_class).float().cpu().numpy()
                if y_pred is None:
                    y_pred = np.copy(pred_label)
                    y_true = np.copy(true_label)
                    y_score = np.copy(pred_score)
                    y_one_hot_np = y_one_hot
                else:
                    y_pred = np.concatenate((y_pred, pred_label))
                    y_true = np.concatenate((y_true, true_label))
                    y_score = np.concatenate((y_score, pred_score))
                    y_one_hot_np = np.concatenate((y_one_hot_np, y_one_hot))
                
                
                val_acc.append((pred_label == true_label).mean())
                val_ap.append(1)

    print(val_flag)

    if time_prediction:
        print("NLL Loss  ", NLL_total / num_test_instance)
        print("MSE Loss  ", MSE_total / num_test_instance)
        print("MAE Loss  ", MAE_total / num_test_instance)
        return NLL_total, MSE_total, MAE_total, num_test_instance, time_predicted_total, time_gt_total
    else:
        cm = confusion_matrix(y_true, y_pred)
        print('Confusion Matrix')
        print(cm)
        if (interpretation) and (val_flag == 'test'):
            _, _, result = process_pattern(walk_pattern_total, walk_score_total, pattern_dict=model.position_encoder.pattern, non_idx=model.num_layers*2, pattern_label=walk_pattern_label_total)
            print('result')
            print(result)
            print('walk pattern')
            print(model.position_encoder.pattern)
            print('walk pattern number:', len(np.unique(walk_pattern_total)))
        val_auc = roc_auc_score(y_one_hot_np, y_score)

        return np.mean(val_acc), np.mean(val_ap), None, val_auc, cm
        