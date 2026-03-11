# -*- coding: utf-8 -*-
"""
Free-Form (flexible) prediction with retraining strategy for STD2Vformer.

Workflow:
  Phase 1 – Pretrain: train the model with full pred_len (fixed horizon).
  Phase 2 – Finetune: for each target length in pred_len_test, reload the
             pretrained weights and fine-tune a dedicated model.
  Test:      for each pred_len_test, load the corresponding finetuned model
             and evaluate.
"""

import os
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from datetime import datetime

from model import *
from torch_utils.load_wave_graph import *
from data.dataset import split_dataset
import torch_utils as tu
from torch_utils import Write_csv, earlystopping
from data.data_process import *
from data.get_data import build_dataloader
import test

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EXP_retraining():
    """
    Free-Form prediction via a two-phase retrain strategy.

    Phase 1 (pretrain):  train with the full `pred_len` horizon.
    Phase 2 (finetune):  for each length in `pred_len_test`, load the
                         pretrained weights and fine-tune a dedicated model.

    Only STD2Vformer is supported in this class; for all other models use EXP.
    """

    def __init__(self, args):
        assert args.resume_dir == args.output_dir, \
            'resume_dir and output_dir must match'
        if args.output_dir in (None, 'None', 'none'):
            args.output_dir = None
            tu.config.create_output_dir(args)
            args.resume_dir = args.output_dir

        # Retrain mode: each stage trains/tests with a fixed horizon.
        # The flexible flag is kept False so that get_data uses pred_len directly.
        self.args = args
        self.args.flexible = False
        self.mode = 'train'   # 'train' = pretrain phase, 'finetune' = finetune phase
        self.train_time = 0.0
        self.peak_memory = 0.0

        tu.dist.init_distributed_mode(args)
        self.build_dataloader(args)
        self.build_model(args, self.adj)
        self.build_optim()
        self.build_loss()
        self.build_path(args)

        if args.resume and args.train:
            self._resume(args)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _resume(self, args):
        """Load the latest available checkpoint (finetune > pretrain)."""
        print('Loading pretrained checkpoint for resume...')
        try:
            dp_mode = args.args.dp_mode
        except AttributeError:
            dp_mode = True

        resume_base = os.path.join(args.resume_dir, args.model_name)
        if not os.path.exists(resume_base):
            return
        pkl_list = os.listdir(resume_base)

        # Priority: most-recent finetune → pretrain
        candidates = [f'predlen_{pl}' for pl in reversed(args.pred_len_test)]
        candidates.append('pretrain_model')
        for stage in candidates:
            matches = [p for p in pkl_list if stage in p]
            if matches:
                path = os.path.join(resume_base, matches[0])
                self.load_best_model(path=path, args=args, distributed=dp_mode)
                self.build_path(args)
                break

    def build_path(self, args):
        """Set up the checkpoint path and EarlyStopping for the current stage."""
        output_path = os.path.join(args.output_dir, args.model_name)
        os.makedirs(output_path, exist_ok=True)

        if self.mode == 'train':
            filename = args.data_name + '_best_pretrain_model.pkl'
        else:
            filename = args.data_name + f'_best_model_predlen_{args.pred_len}.pkl'

        path = os.path.join(output_path, filename)
        self._save_path = path
        self.output_path = output_path
        print(f'[EXP_retraining] Checkpoint path: {path}')

        self.early_stopping = earlystopping.EarlyStopping(
            path=path, optimizer=self.optimizer,
            scheduler=self.lr_optimizer, patience=args.patience
        )

    def build_optim(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )
        self.optimizer = optimizer
        lr_optimizer = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.args.end_epoch, eta_min=self.args.lr / 1000
        )
        self.lr_optimizer = lr_optimizer

    def build_loss(self):
        self.criterion = nn.L1Loss()
        self.criterion2 = nn.MSELoss()

    def build_dataloader(self, args):
        (self.adj,
         self.train_dataloader, self.val_dataloader, self.test_dataloader,
         self.train_sampler, self.val_sampler, self.test_sampler
         ) = build_dataloader(args)

    def build_model(self, args, adj):
        if args.model_name != 'STD2Vformer':
            raise NotImplementedError(
                f'EXP_retraining only supports STD2Vformer; got {args.model_name}. '
                'Use EXP for other models.'
            )
        args.dropout = 0.0
        self.model = STD2Vformer(
            in_feature=args.num_features, num_nodes=args.num_nodes,
            adj=adj, dropout=args.dropout, args=args
        )
        self.model.to(device)
        self.model = tu.dist.ddp_model(self.model, [args.local_rank])
        if args.dp_mode:
            self.model = nn.DataParallel(self.model)
            print('using dp mode')

    # ------------------------------------------------------------------
    # Core training loop
    # ------------------------------------------------------------------

    def train_test_one_epoch(self, args, dataloader, adj, save_manager,
                             epoch, mode='train'):
        if mode == 'train':
            self.model.train()
            self.optimizer.zero_grad()
        elif mode in ('test', 'val'):
            self.model.eval()
        else:
            raise NotImplementedError

        metric_logger = tu.metric.MetricMeterLogger()

        for index, unpacked in enumerate(
                metric_logger.log_every(dataloader, header=mode,
                                        desc=f'{mode} epoch {epoch}')):
            seqs, seqs_time, targets, targets_time = unpacked
            seqs, targets = seqs.cuda().float(), targets.cuda().float()
            seqs_time, targets_time = seqs_time.cuda().float(), targets_time.cuda().float()
            seqs, targets = seqs.permute(0, 2, 3, 1), targets.permute(0, 2, 3, 1)
            seqs_time, targets_time = (seqs_time.permute(0, 2, 3, 1),
                                       targets_time.permute(0, 2, 3, 1))

            adj_arr = np.array(adj)
            pred = self.model(seqs, adj_arr, seqs_time=seqs_time,
                              targets_time=targets_time, targets=targets,
                              mode=mode, index=index, epoch=epoch)

            if mode == 'train':
                pred, loss_part = pred[0], pred[1]
            else:
                pred, loss_part = pred, 0

            targets = targets[:, 0:1, ...]
            if pred.shape[1] != 1:
                pred = pred[:, 0:1, ...]
            assert pred.shape == targets.shape

            loss1 = self.criterion(pred.to(targets.device), targets)
            loss2 = self.criterion2(pred.to(targets.device), targets)
            if args.loss_type == 'MAE':
                loss = loss1
            elif args.loss_type == 'MSE':
                loss = loss2
            elif args.loss_type == 'smooth_l1_loss':
                loss = F.smooth_l1_loss(pred.to(targets.device), targets)
            else:
                loss = loss1 + 0.3 * loss2

            mse = torch.mean(torch.sum((pred - targets) ** 2, dim=1).detach())
            mae = torch.mean(torch.sum(torch.abs(pred - targets), dim=1).detach())
            metric_logger.update(loss=loss, mse=mse, mae=mae)

            step_logs = metric_logger.values()
            step_logs['epoch'] = epoch
            save_manager.save_step_log(mode, **step_logs)

            if mode == 'train':
                (loss + loss_part).sum().backward()
                if args.clip_max_norm > 0:
                    nn.utils.clip_grad.clip_grad_norm_(
                        self.model.parameters(), args.clip_max_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

        epoch_logs = metric_logger.get_finish_epoch_logs()
        epoch_logs['epoch'] = epoch
        save_manager.save_epoch_log(mode, **epoch_logs)
        return epoch_logs

    def _run_stage(self, args, stage_label):
        """Run one complete training stage and load the best checkpoint."""
        start_epoch = getattr(self, 'start_epoch', 0) if args.resume else 0
        print(f'[{stage_label}] pred_len={args.pred_len}, start_epoch={start_epoch}')

        save_manager = tu.save.SaveManager(
            args.output_dir, args.model_name, 'mse',
            compare_type='lt', ckpt_save_freq=30
        )
        save_manager.save_hparam(args)

        # Optional GPU memory tracking
        if PYNVML_AVAILABLE:
            pynvml.nvmlInit()
            cuda_ids = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')
            handles = [pynvml.nvmlDeviceGetHandleByIndex(int(i)) for i in cuda_ids]
            peak_mem = sum(pynvml.nvmlDeviceGetMemoryInfo(h).used for h in handles)
        else:
            peak_mem = 0

        t_start = time.time()
        for epoch in range(start_epoch, args.end_epoch):
            if tu.dist.is_dist_avail_and_initialized():
                self.train_sampler.set_epoch(epoch)
                self.val_sampler.set_epoch(epoch)
            tu.dist.barrier()

            self.train_test_one_epoch(
                args, self.train_dataloader, self.adj, save_manager, epoch, mode='train')
            self.lr_optimizer.step()
            val_logs = self.train_test_one_epoch(
                args, self.val_dataloader, self.adj, save_manager, epoch, mode='val')

            self.early_stopping(val_logs['mse'], model=self.model, epoch=epoch)
            if self.early_stopping.early_stop:
                break

        self.train_time = time.time() - t_start

        if PYNVML_AVAILABLE:
            peak_mem = max(peak_mem,
                           sum(pynvml.nvmlDeviceGetMemoryInfo(h).used for h in handles))
            pynvml.nvmlShutdown()
        self.peak_memory = peak_mem / 1024 / 1024

        # Load the best checkpoint from this stage
        try:
            dp_mode = args.args.dp_mode
        except AttributeError:
            dp_mode = True
        self.load_best_model(path=self._save_path, args=args, distributed=dp_mode)

    # ------------------------------------------------------------------
    # Public train() and test()
    # ------------------------------------------------------------------

    def train(self):
        args = self.args

        # -------- Phase 1: Pretrain --------
        if self.mode == 'train':
            print('=' * 60)
            print('Phase 1: Pretraining with full pred_len =', args.pred_len)
            print('=' * 60)
            tu.config.create_output_dir(args)
            self.build_path(args)
            self._run_stage(args, 'Pretrain')

        # -------- Phase 2: Finetune per pred_len_test --------
        print('=' * 60)
        print('Phase 2: Finetuning for each pred_len_test')
        print('=' * 60)
        self.mode = 'finetune'
        for pl in args.pred_len_test:
            print(f'--- Finetuning  pred_len = {pl} ---')
            ft_args = deepcopy(args)
            ft_args.pred_len = pl
            self.build_dataloader(ft_args)
            self.build_optim()          # fresh optimiser for each finetune stage
            self.build_path(ft_args)
            self._run_stage(ft_args, f'Finetune-{pl}')

    def ddp_module_replace(self, param_ckpt):
        return {k.replace('module.', ''): v.cpu() for k, v in param_ckpt.items()}

    def load_best_model(self, path, args=None, distributed=True):
        if not os.path.exists(path):
            print(f'Checkpoint {path} not found – using current weights.')
            return
        ckpt = torch.load(path)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.lr_optimizer.load_state_dict(ckpt['lr_scheduler'])
        self.start_epoch = ckpt.get('epoch', 0)
        self.train_time = ckpt.get('train_time', self.train_time)
        self.peak_memory = ckpt.get('memory', self.peak_memory)

    def test(self, **kwargs):
        pred_len_test = kwargs.get('pred_len_test', self.args.pred_len)
        print(f'[EXP_retraining.test] pred_len_test = {pred_len_test}')

        test_args = deepcopy(self.args)
        test_args.flexible = False
        test_args.pred_len = pred_len_test

        try:
            dp_mode = self.args.args.dp_mode
        except AttributeError:
            dp_mode = True

        # Load the finetuned model for this specific horizon
        ckpt_path = os.path.join(
            self.output_path,
            test_args.data_name + f'_best_model_predlen_{pred_len_test}.pkl'
        )
        self.build_dataloader(test_args)
        # Rebuild model with the correct pred_len context (safe for STD2Vformer)
        self.build_model(test_args, self.adj)
        self.build_optim()
        self.load_best_model(path=ckpt_path, args=test_args, distributed=dp_mode)

        star = datetime.now()
        metric_dict = test.test(test_args, self.model,
                                test_dataloader=self.test_dataloader,
                                adj=self.adj)
        test_cost_time = (datetime.now() - star).total_seconds()
        print(f'Test cost: {test_cost_time:.2f}s')

        mae  = metric_dict['mae']
        mse  = metric_dict['mse']
        rmse = metric_dict['rmse']
        mape = metric_dict['mape']

        os.makedirs('./results/', exist_ok=True)
        log_path = './results/experimental_logs.csv'
        if not os.path.exists(log_path):
            table_head = [['dataset', 'model', 'time', 'LR', 'batch_size', 'seed',
                           'train_time', 'peak_memory',
                           'MAE', 'MSE', 'RMSE', 'MAPE',
                           'seq_len', 'pred_len', 'pred_len_test',
                           'd_model', 'd_ff', 'M', 'D2V_outmodel',
                           'test_cost_time', 'Loss',
                           'flexible', 'retrain', 'info', 'output_dir']]
            Write_csv.write_csv(log_path, table_head, 'w+')

        time_str = datetime.now().strftime('%Y%m%d-%H%M%S')
        a_log = [{'dataset':        self.args.data_name,
                  'model':          self.args.model_name,
                  'time':           time_str,
                  'LR':             self.args.lr,
                  'batch_size':     self.args.batch_size,
                  'seed':           self.args.seed,
                  'train_time':     self.train_time,
                  'peak_memory':    self.peak_memory,
                  'MAE':            mae,
                  'MSE':            mse,
                  'RMSE':           rmse,
                  'MAPE':           mape,
                  'seq_len':        self.args.seq_len,
                  'pred_len':       self.args.pred_len,
                  'pred_len_test':  pred_len_test,
                  'd_model':        self.args.d_model,
                  'd_ff':           self.args.d_ff,
                  'M':              self.args.M,
                  'D2V_outmodel':   self.args.D2V_outmodel,
                  'test_cost_time': test_cost_time,
                  'Loss':           self.args.loss_type,
                  'flexible':       True,
                  'retrain':        True,
                  'info':           self.args.info,
                  'output_dir':     self.args.output_dir}]
        Write_csv.write_csv_dict(log_path, a_log, 'a+')
