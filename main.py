# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.chdir('')
import io
import sys
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
import torch_utils as tu
from torch_utils.set_seed import set_seed
import torch
from exp_.exp import EXP
from exp_.exp_retraining import EXP_retraining

#————————————————————————————————————————————————————————————————————————————————————————————————————————————
# #########################################!!! Attention !!! #################################################
# Distributed training means running on different machines, non-distributed training means running on one machine. (Multiple cards on one machine does not count as distributed training)
# Note that when you create a new Tensor, if you create it at init time, you should use nn.Parameter() so that the Tensor will be put on the correct cuda.
# Even if the same model, but different dataset, the corresponding output path should be different, otherwise the new code will delete all files under the current path when running (within the train function).
# If there is no batch dimension before inputting the data into the model, then convert its type to numpy, otherwise the first dimension will be split in half (two cards) by default.
# In the build model when the input adj, at this time the adj is already normalized Laplace matrix
#—————————————————————————————————————————————————————————————————————————————————————————————————————————————

device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v == 'True':
        return True
    if v == 'False':
        return False

def str2list(v):
    if isinstance(v, list):
        return v
    v = [int(x) for x in v.strip('[]').split(',')]
    return v

'''The other parameters are in the config file.'''
def add_config(parser):
    parser.add_argument('--exp_name', default='deep_learning', choices=['deep_learning'])
    parser.add_argument('--train', default=True,type=str2bool,choices=[True,False], help='train or not ')
    parser.add_argument('--resume', default=False, type=str2bool, choices=[True, False], help='Load pre-trained model or not')
    parser.add_argument('--visual', default=False, type=str2bool, choices=[True, False], help='Visualization or not')
    parser.add_argument('--output_dir',type=str,default='./experiments/exp30/',help='None will automatically +1 from the existing output file, if specified it will overwrite the existing output file.')
    parser.add_argument('--resume_dir', type=str,default='./experiments/exp30/',help='Retrieve the location of the checkpoint')
    parser.add_argument('--dp_mode', type=str2bool, default=False,help='Does it run on multiple cards')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=16)
    # model settings
    parser.add_argument('--model_name', type=str, default='STD2Vformer',help=['STD2Vformer'])
    parser.add_argument('--gnn_type', type=str, default='dcn', choices=['dcn', 'gat', 'gcn'])
    parser.add_argument('--data_name', type=str, default='METR-LA', choices=['PeMS-Bay','METR-LA','PEMS04','PEMS08'])

    # dataset settings
    parser.add_argument('--seq_len', type=int, default=12)
    parser.add_argument('--pred_len', type=int, default=3, help='Training prediction horizon')
    # ---- Free-Form (flexible) prediction ----
    parser.add_argument('--flexible', type=str2bool, default=False,
                        help='Enable Free-Form prediction: test at arbitrary horizons without a fixed pred_len')
    parser.add_argument('--retrain', type=str2bool, default=False,
                        help='Flexible mode only: finetune a dedicated model per pred_len_test (True) '
                             'vs. single model evaluated at multiple horizons (False)')
    parser.add_argument('--pred_len_test', type=str2list, default=[3],
                        help='Prediction horizons to evaluate at test time (e.g. [3,6,12]); '
                             'only used when flexible=True')
    parser.add_argument('--alpha', type=float, default=0.0,
                        help='Probability of randomly shortening the target during training '
                             'as data augmentation for flexible prediction (0 = disabled)')
    # ---- Non-blind meta-information ----
    parser.add_argument('--is_no_blind', type=str2bool, default=False,
                        help='Inject non-blind meta information (pred_len/seq_len ratio) into the '
                             'model so it is aware of the prediction horizon at inference time')
    # Settings related to model parameters
    parser.add_argument('--num_features', help="Input model's feature dimensions(will be automatically specified based on the dataset)")
    parser.add_argument('--time_features',
                        help='Time step features of the input model (will be specified automatically based on the dataset)')
    parser.add_argument('--num_nodes',
                        help='Input the number of model nodes (will be specified automatically based on the dataset)')
    parser.add_argument('--d_model', type=int, default=64, help='Hidden layer dimension 1')
    parser.add_argument('--d_ff', type=int, default=128, help='Hidden layer dimension 2')
    parser.add_argument('--num_gcn', type=int, default=10, help='Number of GCNs')
    parser.add_argument('--patch_len', type=int, default=3, help='length of patch_len')
    parser.add_argument('--stride', type=int, default=1, help='length of stride')
    parser.add_argument('--points_per_hour', type=int, default=12, help='How many data sampling points in an hour (related to the dataset)')

    # D2V setting
    parser.add_argument('--D2V_outmodel', type=int, default=64, help=' Indicates the number of corner frequencies used by D2V')
    parser.add_argument('--M', type=int, default=8, help=' means take top-k nodes')

    parser.add_argument('--loss_type', type=str, default='smooth_l1_loss', choices=['MAE','MSE','smooth_l1_loss'])
    parser.add_argument('--info', type=str, default='None', help='Experimental information')
    return parser

def preprocess_args(args):
    args.pin_memory = False # The way read data is accelerated in Dataloader
    return args

if __name__ == '__main__':
    args = tu.config.get_args(add_config) # Getting set hyperparameters
    args = preprocess_args(args)
    set_seed(args.seed) # Set random seed

    print(f"|{'=' * 101}|")
    # Use the __dict__ method to get a dictionary of arguments and traverse the dictionary afterwards
    for key, value in args.__dict__.items():
        # Because the arguments may not all be str, it is necessary to convert all the data to str
        print(f"|{str(key):>50s}|{str(value):<50s}|")
    print(f"|{'=' * 101}|")
    print(device)
    # Select experiment class based on prediction mode
    if args.flexible and args.retrain:
        exp = EXP_retraining(args)
    elif args.exp_name == 'deep_learning':
        exp = EXP(args)
    else:
        raise ValueError('No exp file with name {0}'.format(args.exp_name))

    torch.autograd.set_detect_anomaly(True)
    if args.train:
        exp.train()
    with torch.no_grad():
        if args.flexible:
            # Evaluate at each requested prediction horizon
            for pred_len_test in args.pred_len_test:
                exp.test(pred_len_test=pred_len_test)
        else:
            exp.test()


