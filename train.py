import torch
import numpy as np
from tqdm import tqdm
import math
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
# from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from eval import *
import logging
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.ticker').disabled = True
import sys
import histogram
import os

def train_val(dataset, model, mode, bs, epochs, criterion, optimizer, early_stopper, ngh_finders, logger, interpretation=False, time_prediction=False):
    partial_ngh_finder, full_ngh_finder = ngh_finders
    device = model.n_feat_th.data.device
    num_instance = dataset.get_size()
    num_batch = math.ceil(num_instance / bs)
    dataset.set_batch_size(bs)
    
    logger.info('num of training instances: {}'.format(num_instance))
    logger.info('num of batches per epoch: {}'.format(num_batch))
    model.test = False
    

    if interpretation:
        roc_auc_score = utils.roc_auc_score_single
    else:
        roc_auc_score = utils.roc_auc_score_multi

    for epoch in range(epochs):
        model.update_ngh_finder(partial_ngh_finder)
        dataset.initialize()
        dataset.reset()
        acc, ap, f1, auc, m_loss = [], [], [], [], []
        logger.info('start {} epoch'.format(epoch))
        NLL_total = None
        
        y_true, y_pred, y_one_hot_np = None, None, None

        for k in tqdm(range(int(num_batch))):
            src_1_l_cut, src_2_l_cut, dst_l_cut, ts_l_cut, e_l_cut, true_label = dataset.train_samples()
            
            model.train()
            optimizer.zero_grad()
            if time_prediction:
                true_label_torch = torch.from_numpy(true_label).to(device)
                _pred_score, _ = model.contrast(src_1_l_cut, src_2_l_cut, dst_l_cut, ts_l_cut, e_l_cut, endtime_pos=true_label_torch)   # the core training code
                ave_log_t, pred_score, _ = _pred_score
            else:
                true_label_torch = torch.from_numpy(true_label).long().to(device)
                pred_score, _ = model.contrast(src_1_l_cut, src_2_l_cut, dst_l_cut, ts_l_cut, e_l_cut)   # the core training code
            

            if time_prediction:
                loss = pred_score
            else:
                loss = criterion(pred_score, true_label_torch)
            
            loss.backward()
            optimizer.step()

            
            # collect training results
            with torch.no_grad():
                model.eval()
                if time_prediction:
                    if NLL_total is None:
                        NLL_total = pred_score
                        MSE = ave_log_t
                    else:
                        NLL_total += pred_score
                        MSE += ave_log_t
                else:
                    pred_label = torch.argmax(pred_score, dim=1).cpu().detach().numpy()
                    acc.append((pred_label == true_label).mean())
                    ap.append(1)
                    m_loss.append(loss.item())
                    y_one_hot = torch.nn.functional.one_hot(torch.from_numpy(true_label).long(), num_classes=model.num_class).float().cpu().numpy()
            
                    if y_pred is None:
                        y_pred = np.copy(pred_label)
                        y_true = np.copy(true_label)
                        y_one_hot_np = y_one_hot
                        pred_score_np = torch.nn.functional.softmax(pred_score, dim=1).cpu().numpy()
                    else:
                        y_pred = np.concatenate((y_pred, pred_label))
                        y_true = np.concatenate((y_true, true_label))
                        y_one_hot_np = np.concatenate((y_one_hot_np, y_one_hot))
                        pred_score_np = np.concatenate((pred_score_np, torch.nn.functional.softmax(pred_score, dim=1).cpu().numpy()))
        if time_prediction:
            print("train")
            print(NLL_total/dataset.get_size())
            print(MSE/dataset.get_size())
        else:
            print("train")
            cm = confusion_matrix(y_true, y_pred)
            print(cm)
            logger.info('confusion matrix: ')
            logger.info(', '.join(str(r) for r in cm.reshape(1,-1)))

            acc = np.mean(acc)
            auc = roc_auc_score(y_one_hot_np, pred_score_np)
        
        if time_prediction:
            
            NLL_loss, num, time_predicted_total, time_gt_total = eval_one_epoch('val for {} nodes'.format(mode), model, dataset, val_flag='val',interpretation=interpretation, time_prediction=time_prediction)
            logger.info('val NLL: {}  Number: {}'.format(NLL_loss, num))
            val_auc = -NLL_loss.cpu().numpy()
        else:
            val_acc, val_ap, val_f1, val_auc, cm = eval_one_epoch('val for {} nodes'.format(mode), model, dataset, val_flag='val',interpretation=interpretation, time_prediction=time_prediction)
            logger.info('confusion matrix: ')
            logger.info(', '.join(str(r) for r in cm.reshape(1,-1)))
        model.update_ngh_finder(full_ngh_finder)
        if time_prediction:
            test_NLL, num, time_predicted_total, time_gt_total = eval_one_epoch('test for {} nodes'.format(mode), model, dataset, val_flag='test',interpretation=interpretation, time_prediction=time_prediction)
            time_predicted_total = np.exp(time_predicted_total)
            time_gt_total = np.exp(time_gt_total)
            # file_addr = './Histogram/'+dataset.DATA+'-'+str(dataset.time_prediction_type)+'/'
            # if not os.path.exists(file_addr):
            #     os.makedirs(file_addr)
            
            # with open(file_addr+'time_prediction_histogram'+str(epoch), 'wb') as f:
            #     np.save(f, np.array([time_predicted_total, time_gt_total]))
            # histogram.plot_hist_multi([time_predicted_total, time_gt_total], bins=50, figure_title='Time Prediction Histogram'+str(epoch), file_addr=file_addr, label=['Ours', 'Groundtruth'])
            
            logger.info('test NLL: {}'.format(test_NLL))
        else:
            val_acc_t, val_ap_t, val_f1_t, val_auc_t, cm = eval_one_epoch('val for {} nodes'.format(mode), model, dataset, val_flag='test',interpretation=interpretation, time_prediction=time_prediction)
            logger.info('confusion matrix: ')
            logger.info(', '.join(str(r) for r in cm.reshape(1,-1)))
            logger.info('epoch: {}:'.format(epoch))
            logger.info('epoch mean loss: {}'.format(np.mean(m_loss)))
            logger.info('train acc: {}, val acc: {}, test acc: {}'.format(np.mean(acc), val_acc, val_acc_t))
            logger.info('train auc: {}, val auc: {}, test auc: {}'.format(np.mean(auc), val_auc, val_auc_t))
            logger.info('train ap: {}, val ap: {}, test ap: {}'.format(np.mean(ap), val_ap, val_ap_t))

        # # early stop check and checkpoint saving
        if early_stopper.early_stop_check(val_auc):
            logger.info('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
            logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
            best_checkpoint_path = model.get_checkpoint_path(early_stopper.best_epoch)
            model.load_state_dict(torch.load(best_checkpoint_path))
            logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
            model.eval()
            break
        else:
            torch.save(model.state_dict(), model.get_checkpoint_path(epoch))


