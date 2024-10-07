import numpy as np
import torch
from matplotlib import pyplot as plt
import yaml
import os
import argparse
import torch_utils as tu
from tqdm import tqdm
from sklearn.metrics import r2_score, explained_variance_score,mean_absolute_percentage_error

torch.multiprocessing.set_sharing_strategy('file_system')

def np_eps(data, eps=1e-4):
    data[np.where(np.abs(data) < eps)] = eps
    return data


def smape_m(sources, outputs):
    mae=np.abs(sources-outputs)
    sources_=np.abs(sources)
    outputs_=np.abs(outputs)
    return np.mean(2*mae/(np_eps(sources_)+np_eps(outputs_)))



def mse_m(labels,preds, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mse = np.square(np.subtract(preds, labels)).astype('float32')
        mse = np.nan_to_num(mse * mask)
        return np.mean(mse)


def mae_m(labels,preds, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(preds, labels)).astype('float32')
        mae = np.nan_to_num(mae * mask)
        return np.mean(mae)


def mape_m(labels,preds, null_val=np.nan):
    tmp1=np.mean(labels)*0.1
    mape=np.abs(preds.flatten()- labels.flatten())/((labels+tmp1).flatten())
    return np.mean(mape)

def r2_m(sources, outputs):
    mse=np.square(sources - outputs)
    y_var=np.var(sources)
    y_var=1e-4 if y_var==0 else y_var
    return 1-np.mean(mse/y_var)


def explained_variance_score_np(preds, labels):
    preds = preds.flatten()
    labels = labels.flatten()
    return explained_variance_score(labels, preds)

def calc_metrics(sources, outputs,mean=None,std=None,max=None,min=None):
    '''outputs、sources:(total_len,C=1,num_nodes,pred_len)'''
    assert sources.shape==outputs.shape
    if np.all(mean!=None) and np.all(std!=None):
        if np.isnan(mean).all()==False and np.isnan(std).all()==False:
            if outputs.shape[1]==1:
                mean=mean[:,0:1,:] # TODO Here's taking a dimension of features
                std = std[:, 0:1, :]
            sources=sources*std+mean
            outputs=outputs*std+mean
        else:
            assert print('mean or std has the value nan')
    elif np.all(max!=None) and np.all(min!=None):
        if np.isnan(max).all()==False and np.isnan(min).all()==False:
            if outputs.shape[1]==1:
                min=min[:,0:1,:]
                max = max[:, 0:1, :] # TODO Here's taking a dimension of features

            sources=(sources+1)/2*(max-min)+min
            outputs=(outputs+1)/2*(max-min)+min
        else:
            assert print("max or min has the value nan")
    mse = mse_m(sources, outputs)
    mae = mae_m(sources, outputs)
    rmse = np.sqrt(mse)
    r2 = r2_m(sources, outputs)
    mape = mape_m(sources, outputs)
    smape = smape_m(sources, outputs)
    metric_dict = {
        'mse': mse.item(),
        'mae': mae.item(),
        'rmse': rmse.item(),
        'r2': r2.item(),
        'mape': mape.item(),
        'smape': smape.item(),
    }

    return metric_dict


@torch.no_grad()
def pred_st_graph_data(model, dataloader,adj):
    '''
    :param model: spatio-temporal graph model in cuda
    :param dataloader: torch.utils.data.dataloader which shuffle and drop_last are 'False'
    :return: sources: (total_L, C, N, pred_len), outputs: (total_L, C, N, pred_len)
    total_L is the number of instances.
    '''
    sources = []
    outputs = []

    _iter = tqdm(dataloader)
    for seqs,seqs_time,targets,targets_time in _iter:
        seqs, targets = seqs.cuda().float(), targets.cuda().float()
        seqs_time, targets_time = seqs_time.cuda().float(), targets_time.cuda().float()
        seqs, targets = seqs.permute(0, 2, 3, 1), targets.permute(0, 2, 3, 1)
        seqs_time, targets_time = seqs_time.permute(0, 2, 3, 1), targets_time.permute(0, 2, 3, 1)
        # Model inputs and outputs are (B,C,N,L)
        seqs = seqs.cuda()
        pred = model(seqs,adj,seqs_time=seqs_time,targets_time=targets_time)
        if pred.shape[1]!=1:
            pred=pred[:,0:1,...]
        # print(pred.shape)
        # TODO By default the metrics are calculated for the first dimensional feature
        sources.extend(targets[:,0:1,:,:].detach().cpu().numpy())
        outputs.extend(pred.detach().cpu().numpy())
    sources = np.array(sources)
    outputs = np.array(outputs)

    return sources, outputs


def test(args,model,test_dataloader,adj):
    # tu.model_tool.seed_everything(args.seed, benchmark=False)

    for f in os.listdir(args.resume_dir):
        if f.startswith('_best_'):
            print('best checkpoint:{}'.format(f))
    print('args : {}'.format(args))

    model.eval()
    test_out_dir=os.path.join(args.output_dir,'test')
    os.makedirs(test_out_dir, exist_ok=True) # Flush the GPU's cache

    sources, outputs = pred_st_graph_data(model, test_dataloader,adj)
    torch.cuda.empty_cache()

    if args.data_name=="PeMS-Bay"or args.data_name=="METR-LA":
        mean=test_dataloader.mean # Get the corresponding standardized mean and variance for back-standardization
        std=test_dataloader.std
        metric_dict = calc_metrics(sources, outputs,mean=mean,std=std)
    elif args.data_name == "PEMS04" or args.data_name == "PEMS08":
        min=test_dataloader.min
        max=test_dataloader.max
        metric_dict = calc_metrics(sources, outputs,min=min,max=max)
    else:
        assert print('Dataset normalization undefined')

    for k, v in metric_dict.items():
        metric_dict[k] = round(v, 5)  # round to the nearest integer.

    print(metric_dict)  # print the result

    with open(os.path.join(test_out_dir, '{}_metric.yaml'.format(args.train)), 'w+') as f:
        f.write(yaml.dump(metric_dict))

    # visualize_plt(sources, outputs, test_out_dir, args.train_test, max_show=args.max_show)
    return metric_dict
