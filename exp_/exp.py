from model import *
from torch_utils.load_wave_graph import *
from data.dataset import split_dataset
import torch_utils as tu
from torch_utils import Write_csv, earlystopping
from data.data_process import *
from data.get_data import build_dataloader
import torch
import torch.nn as nn
import numpy as np
import test
import yaml
from datetime import datetime

device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')
class EXP():
    def __init__(self,args):
        assert args.resume_dir==args.output_dir
        self.agrs=args
        tu.dist.init_distributed_mode(args)  # Initializing Distributed Training

        # get_data
        (adj, self.train_dataloader,self.val_dataloader, self.test_dataloader,
         self.train_sampler,self.val_sampler,self.test_sampler) = build_dataloader(args)
        self.adj = adj  # adjacency matrix

        # get_model
        self.build_model(args, adj)
        self.model.to(device)

        self.model = tu.dist.ddp_model(self.model, [args.local_rank])  # DDP
        if args.dp_mode:
            self.model = nn.DataParallel(self.model)  # DP
            print('using dp mode')

        # Something for training
        self.criterion=nn.L1Loss()
        self.criterion2 = nn.MSELoss()

        # Adam introducing weight-decaying
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  #
        self.optimizer=optimizer

        # weight decay
        lr_optimizer = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.end_epoch,eta_min=args.lr / 1000)
        self.lr_optimizer=lr_optimizer

        if args.output_dir==None or args.output_dir=='None' or args.output_dir=='none':
            args.output_dir = None
            tu.config.create_output_dir(args)  # Creating a directory for output
            args.resume_dir=args.output_dir

        output_path = os.path.join(args.output_dir,args.model_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        path = os.path.join(output_path, args.data_name + '_best_model.pkl')
        self.output_path = output_path
        self.early_stopping = earlystopping.EarlyStopping(path=path, optimizer=self.optimizer,
                                                          scheduler=self.lr_optimizer, patience=args.patience)
        resume_path = os.path.join(args.resume_dir,args.model_name)
        if not os.path.exists(resume_path):
            raise print('No corresponding path to read pre-trained weights was found')
        resume_path = os.path.join(resume_path, args.data_name + '_best_model.pkl')
        self.resume_path = resume_path

        if args.resume:
            print('Load the pre-trained checkpoint')
            try:
                dp_mode = args.args.dp_mode
            except AttributeError as e:
                dp_mode = True
            hparam_path = os.path.join(args.output_dir, 'hparam.yaml')
            with open(hparam_path, 'r') as f:
                hparam_dict = yaml.load(f, yaml.FullLoader)
                args.output_dir = hparam_dict['output_dir']

            # Load the best checkpoint
            self.load_best_model(path=self.resume_path,args=args, distributed=dp_mode)

    '''Build model'''
    def build_model(self,args,adj):
        if args.model_name == 'STD2Vformer':
            args.dropout = 0.0
            args.D2V_outmodel = args.D2V_outmodel  # Denote the output of D2V as dimension
            args.M = args.M   # Indicates how many nodes are taken
            self.model = STD2Vformer(in_feature=args.num_features, num_nodes=args.num_nodes, adj=adj, dropout=args.dropout,
                                args=args)

        else:
            raise NotImplementedError

    '''Code under an epoch'''
    def train_test_one_epoch(self,args,dataloader,adj,save_manager: tu.save.SaveManager,epoch,mode='train',max_iter=float('inf'),**kargs):
        if mode == 'train':
            self.model.train()
            self.optimizer.zero_grad()
        elif mode == 'test' or mode =='val':
            self.model.eval()
        else:
            raise NotImplementedError

        metric_logger = tu.metric.MetricMeterLogger()  # Initialize a dictionary to record the loss results of the corresponding training


        for index, unpacked in enumerate(
                metric_logger.log_every(dataloader, header=mode, desc=f'{mode} epoch {epoch}')):
            if index > max_iter:
                break
            seqs, seqs_time,targets,targets_time = unpacked # (B,L,C,N)
            seqs, targets = seqs.cuda().float(), targets.cuda().float()
            seqs_time, targets_time = seqs_time.cuda().float(), targets_time.cuda().float()
            seqs,targets=seqs.permute(0,2,3,1),targets.permute(0,2,3,1)  #(B,C,N,L)
            seqs_time, targets_time = seqs_time.permute(0, 2, 3, 1), targets_time.permute(0, 2, 3, 1) #(B,C,N,L)
            # TODO Input and output are both (B,C,N,L). The output's feature dimension defaults to 1
            self.adj = np.array(self.adj)  # If it's not an array, then the first dimension is split in half when it's fed into the mod
            pred = self.model(seqs,self.adj,seqs_time=seqs_time,targets_time=targets_time,targets=targets,mode=mode,index=index,epoch=epoch)  # 输入模型

            if args.model_name=='STD2Vformer':
                if mode == 'train':
                    pred,loss_part=pred[0],pred[1]
                else:
                    pred, loss_part = pred, 0

            # Calculate the loss(defaults to calculating the first feature dimension)
            targets=targets[:, 0:1, ...]
            if pred.shape[1]!=1:
                pred=pred[:,0:1,...]

            assert pred.shape == targets.shape
            loss1 = self.criterion(pred.to(targets.device), targets) # MAE loss
            loss2=self.criterion2(pred.to(targets.device), targets) # MSE loss
            if args.loss_type=='MAE':
                loss = loss1
            elif args.loss_type=='MSE':
                loss = loss2
            else:
                loss = loss1 + 0.3 * loss2

            # Calculate MSE, MAE losses
            mse = torch.mean(torch.sum((pred - targets) ** 2, dim=1).detach())
            mae = torch.mean(torch.sum(torch.abs(pred - targets), dim=1).detach())

            metric_logger.update(loss=loss, mse=mse, mae=mae)  # Update training records

            step_logs = metric_logger.values()
            step_logs['epoch'] = epoch
            save_manager.save_step_log(mode, **step_logs)   # Save the training loss for each batch

            if mode == 'train':
                if args.model_name=='STD2Vformer':
                    loss=loss+loss_part
                loss.sum().backward()
                # grad-crop
                if args.clip_max_norm > 0:  #  Crop value greater than 0
                    nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), args.clip_max_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

        epoch_logs = metric_logger.get_finish_epoch_logs()
        epoch_logs['epoch'] = epoch
        save_manager.save_epoch_log(mode, **epoch_logs)  # Save the training loss for each epoch

        return epoch_logs

    def train(self):
        args=self.agrs
        if args.resume!=True:
            tu.config.create_output_dir(args)  # Create directories for output
            print('output dir: {}'.format(args.output_dir))
            start_epoch = 0
        else:
            start_epoch=self.start_epoch

        # The following hyperparameters are saved
        save_manager = tu.save.SaveManager(args.output_dir, args.model_name, 'mse', compare_type='lt', ckpt_save_freq=30)
        save_manager.save_hparam(args)

        # Start training
        for epoch in range(start_epoch, args.end_epoch):
            if tu.dist.is_dist_avail_and_initialized():
                self.train_sampler.set_epoch(epoch)
                self.val_sampler.set_epoch(epoch)
                self.test_sampler.set_epoch(epoch)

            tu.dist.barrier()

            # train
            self.train_test_one_epoch(args,self.train_dataloader,self.adj, save_manager, epoch, mode='train')

            self.lr_optimizer.step()  # lr decay

            # val
            val_logs = self.train_test_one_epoch(args, self.val_dataloader, self.adj, save_manager, epoch, mode='val')

            # test
            test_logs = self.train_test_one_epoch(args,self.test_dataloader,self.adj, save_manager, epoch,mode='test')


            # early-stopping
            self.early_stopping(val_logs['mse'], model=self.model, epoch=epoch)
            if self.early_stopping.early_stop:
                break
        # Training complete. Read the best weights.
        try:
            dp_mode = args.args.dp_mode
        except AttributeError as e:
            dp_mode = True
        output_path = os.path.join(self.output_path, args.data_name + '_best_model.pkl')
        self.load_best_model(path=output_path, args=args, distributed=dp_mode)


    def ddp_module_replace(self,param_ckpt):
        return {k.replace('module.', ''): v.cpu() for k, v in param_ckpt.items()}

    # TODO Load best checkpoint
    def load_best_model(self, path, args=None, distributed=True):

        ckpt_path = path
        if not os.path.exists(ckpt_path):
            print('The path {0} does not exist and the parameters of the model are randomly initialized'.format(ckpt_path))
        else:
            ckpt = torch.load(ckpt_path)

            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.lr_optimizer.load_state_dict(ckpt['lr_scheduler'])
            self.start_epoch=ckpt['epoch']

    def test(self):
        args=self.agrs
        try:
            dp_mode = args.args.dp_mode
        except AttributeError as e:
            dp_mode = True

        # Load best checkpoint
        if args.resume:
            self.load_best_model(path=self.resume_path, args=args, distributed=dp_mode)
        star = datetime.now()
        metric_dict=test.test(args,self.model,test_dataloader=self.test_dataloader,adj=self.adj)
        end=datetime.now()
        test_cost_time=(end-star).total_seconds()
        print("test cost：{0}s".format(test_cost_time))
        mae=metric_dict['mae']
        mse=metric_dict['mse']
        rmse=metric_dict['rmse']
        mape=metric_dict['mape']

        # Create csv file to record training results
        if not os.path.isdir('./results/'):
            os.mkdir('./results/')

        log_path = './results/experimental_logs.csv'
        if not os.path.exists(log_path):
            table_head = [['dataset', 'model', 'time', 'LR',
                          'batch_size', 'seed', 'MAE', 'MSE', 'RMSE','MAPE','seq_len',
                           'pred_len', 'd_model', 'd_ff','M','D2V_outmodel','test_cost_time',
                            'Loss','info','output_dir']]
            Write_csv.write_csv(log_path, table_head, 'w+')

        time = datetime.now().strftime('%Y%m%d-%H%M%S')  # Get current system time
        a_log = [{'dataset': args.data_name, 'model': args.model_name, 'time': time,
                  'LR': args.lr,
                  'batch_size': args.batch_size,
                  'seed': args.seed, 'MAE': mae, 'MSE': mse,'RMSE':rmse,"MAPE":mape,'seq_len': args.seq_len,
                  'pred_len': args.pred_len,'d_model': args.d_model, 'd_ff': args.d_ff,'M':args.M,'D2V_outmodel':args.D2V_outmodel,
                  'test_cost_time': test_cost_time,'Loss':args.loss_type,
                  'info': args.info,'output_dir':args.output_dir}]
        Write_csv.write_csv_dict(log_path, a_log, 'a+')





