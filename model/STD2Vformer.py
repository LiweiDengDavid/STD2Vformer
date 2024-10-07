import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_utils.graph_process import *
from layers.STD2Vformer_related import *
import torch
import torch.nn as nn


class STD2Vformer(nn.Module):
    def __init__(self,adj,in_feature,num_nodes,dropout=0.05,args=None,**kwargs):
        super().__init__()
        self.args=args
        self.in_feature = in_feature
        self.d_model=args.d_model
        # supports = calculate_laplacian_with_self_loop(adj)
        # self.supports = [supports]
        # self.supports_len = 0
        # if supports is not None:
        #     self.supports_len += len(self.supports)
        # if supports is None:
        #     self.supports = []

        # STID
        self.num_layer=3
        self.num_nodes=args.num_nodes
        self.day_of_week_size = 7  # Pick the n days of the week
        self.time_of_day_size = args.points_per_hour * 24  # How many time-steps a day are recorded

        self.time_in_day_emb = nn.Parameter(torch.empty(self.time_of_day_size, self.args.d_model))
        nn.init.xavier_uniform_(self.time_in_day_emb)
        self.day_in_week_emb = nn.Parameter(torch.empty(self.day_of_week_size, self.args.d_model))
        nn.init.xavier_uniform_(self.day_in_week_emb)
        # embedding layer
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.args.num_features * self.args.seq_len, out_channels=self.args.d_model, kernel_size=(1, 1), bias=True)
        # encoding
        self.hidden_dim = self.args.d_model+self.args.d_model +self.args.d_model + self.args.d_model
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)])
        # regression
        self.regression_layer = nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.args.seq_len*self.args.num_features, kernel_size=(1, 1), bias=True)
        self.bn_hidden=nn.BatchNorm2d(in_feature,affine=True)
        # conv_input
        self.conv_input=nn.Conv2d(self.in_feature,self.in_feature,kernel_size=1)

        # Date2Vec
        self.Date2Vec = Date2Vec(num_nodes=args.num_nodes,in_features=args.num_features,date_feature=args.time_features,D2V_outputmode=args.D2V_outmodel,seq_len=args.seq_len
                                 ,output_feature=args.num_features, d_mark=args.time_features,args=args)

        # Fusion
        self.fusion = Fusion_Module(args=args)

        self.weight_inextend=nn.Parameter(torch.randn(args.M,1),requires_grad=True)

        # BN layer
        self.bn = nn.BatchNorm2d(in_feature, affine=True)
        self.bn_funsion = nn.BatchNorm2d(in_feature, affine=True)

        self.pro_out=nn.Linear(self.d_model+self.args.time_features,self.d_model)
        # Pred Model
        self.glu = GLU(in_features=args.d_model, out_features=args.num_features)

        self.construct_memory()

        # In order to balance the parameters constantly reducing the use of
        self.batch_idx_max=1
        # loss
        self.mae=nn.L1Loss()

    def construct_memory(self):
        # Memory bank
        self.memory_bank = nn.Parameter(torch.randn(self.num_nodes, self.d_model,device='cuda'), requires_grad=True)
        nn.init.xavier_normal_(self.memory_bank)
        self.We1 = nn.Parameter(torch.randn(self.d_model, self.d_model,device='cuda'),
                                          requires_grad=True)  # project memory to embedding
        nn.init.xavier_normal_(self.We1)
        self.We2 = nn.Parameter(torch.randn(self.d_model, self.d_model,device='cuda'),
                                          requires_grad=True)  # project memory to embedding
        nn.init.xavier_normal_(self.We2)


    def Embedding(self,input_data,seq_time):
        hour = (seq_time[:, -2:-1, ...] + 0.5) * 23
        min = (seq_time[:, -1:, ...] + 0.5) * 59
        hour_index = (hour * 60 + min) / (60 / self.args.points_per_hour)
        time_in_day_emb = self.time_in_day_emb[
            hour_index[..., -1].squeeze(1).repeat(1, self.args.num_nodes).type(torch.LongTensor)]  # (B,N,D)
        day = (seq_time[:, 2:3, ...] + 0.5) * (6 - 0)
        day_in_week_emb = self.day_in_week_emb[
            day[..., -1].squeeze(1).repeat(1, self.args.num_nodes).type(torch.LongTensor)]
        # time series embedding
        batch_size, _, num_nodes, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(input_data)
        node_emb = []
        node_emb.append(self.memory_bank.clone().unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))#（B,C,N,L）
        # temporal embeddings
        tem_emb = []
        tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
        tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))
        # concate all embeddings
        hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)
        # encoding
        hidden = self.encoder(hidden)
        # regression
        hidden = self.regression_layer(hidden).squeeze(-1)
        B,D,N=hidden.shape
        hidden=hidden.reshape(B,self.args.num_features,self.args.seq_len,N)
        hidden=hidden.permute(0,1,-1,-2) #(B,C,N,L)
        hidden=self.bn_hidden(hidden)
        return hidden #(B,C,N,L)

    def calculate_rate(self,batch_idx,epoch):
        '''It decreases during training until it is 0'''
        if batch_idx>self.batch_idx_max:
            self.batch_idx_max=batch_idx
        global_step = epoch * self.batch_idx_max + batch_idx
        if epoch==0:
            teacher_forcing_ratio = 1
        else:
            teacher_forcing_ratio= 0.99 ** (global_step/(self.batch_idx_max/5)) # 相当于一个epoch，其对应的值缩小5次
        return teacher_forcing_ratio # 100为初始值，表示最开始的初始值


    def forward(self,input,adj,**kwargs):
        self.node_emb1 = torch.matmul(self.memory_bank.clone(), self.We1)
        self.node_emb2 = torch.matmul(self.memory_bank.clone(), self.We2)
        # get Adj
        self.adj_adp = torch.matmul(self.node_emb1, self.node_emb2.transpose(-1, -2))
        # Normalize the process by restricting its domain of values --> so that softmax does not result in only one value of 1 and the rest are 0
        self.adj_adp=(self.adj_adp-torch.mean(self.adj_adp,dim=-1,keepdim=True))/torch.std(self.adj_adp,dim=-1,keepdim=True)

        # The following operation mainly removes the nodes from itself
        self.adj_adp = torch.softmax(torch.relu(self.adj_adp) - torch.eye(self.num_nodes, device='cuda') * 10**6, dim=-1)
        input = self.bn(input) # Normalization in the feature dimension
        input_data = input[:, range(self.in_feature)]
        x = input_data.clone()
        seq_time = kwargs.get('seqs_time')
        pred_time = kwargs.get('targets_time')
        target=kwargs.get('targets')
        batch_idx = kwargs.get('index')
        epoch=kwargs.get('epoch')

        top_value, index = torch.topk(self.adj_adp.clone(), self.args.M-1, dim=-1)

        hidden = self.Embedding(input_data=input_data.clone(), seq_time=seq_time)
        x= self.bn_funsion(hidden+self.conv_input(x))
        x_extend=x[:,:,index]#(B,C,N,M-1,L)
        x_extend=torch.concat([x.unsqueeze(-2),x_extend],dim=-2)#(B,C,N,M,L) Add own node
        D2V_input = torch.cat([seq_time, pred_time], dim=-1)
        D2V_output = self.Date2Vec(x_extend, D2V_input)#(B,C*D2V_outmode,N,M,L+O)
        D2V_x_date=D2V_output[...,:self.args.seq_len]
        D2V_y_date = D2V_output[..., self.args.seq_len:]

        prediction, A = self.fusion(x_extend, D2V_x_date, D2V_y_date,top_value)  # (B,C,N,L)

        # B,_,_,O=pred_time.shape
        # N=x.shape[-2]
        # prediction=torch.cat([prediction, pred_time.repeat(1,1,N,1)],dim=1)
        # prediction=self.pro_out(prediction.transpose(-1,1)).transpose(-1,1)
        prediction = self.glu(prediction)

        if self.training:
            target_extend=target[:,:,index]
            target_extend = torch.concat([target.unsqueeze(-2), target_extend], dim=-2)
            _,A_true=self.fusion(x_extend, x_extend, target_extend,top_value,mode='true')
            # Here, since A and A_true are softmaxed results, it is straightforward to approximate the value
            loss_part=self.mae(A.clone(),A_true.clone())
            rate=self.calculate_rate(batch_idx, epoch)
            return prediction, (loss_part*100)*rate # is multiplied by the corresponding magnification, and this magnification decreases gradually
        else:
            if self.args.visual:
                # (B,C_mark,1,L+O)
                B, C_mark,_,Len = D2V_input.shape
                D2V_input_reshaped = D2V_input.reshape(B, C_mark, -1)
                data_for_tsne = D2V_input_reshaped.reshape(B, -1).cpu().detach()
                # T-SNE
                tsne = TSNE(n_components=2, random_state=42, perplexity=30)
                data_tsne = tsne.fit_transform(data_for_tsne)

                # Visualization
                plt.figure(figsize=(10, 8))
                plt.scatter(data_tsne[:, 0], data_tsne[:, 1], marker='o')
                plt.title('T-SNE Visualization of D2V_input')
                plt.xlabel('TSNE Component 1')
                plt.ylabel('TSNE Component 2')
                plt.grid()
                plt.savefig('Before.png')

                # (B,C*D2V_outmode,N,M,L+O)
                B,D,N,M,Len=D2V_output.shape
                D2V_output_reshaped = D2V_output[:,0,0,:].reshape(B, D, -1)
                data_for_tsne = D2V_output_reshaped.reshape(B, -1).cpu().detach()
                # T-SNE
                tsne = TSNE(n_components=2, random_state=42,perplexity=30)
                data_tsne = tsne.fit_transform(data_for_tsne)

                # Visualization
                plt.figure(figsize=(10, 8))
                plt.scatter(data_tsne[:, 0], data_tsne[:, 1], marker='o')
                plt.title('T-SNE Visualization of D2V_output')
                plt.xlabel('TSNE Component 1')
                plt.ylabel('TSNE Component 2')
                plt.grid()
                plt.savefig('After.png')
                self.args.visual=False

            return prediction





