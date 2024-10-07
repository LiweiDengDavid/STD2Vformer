import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class Fusion_Module(nn.Module):
    def __init__(self,args):
        super(Fusion_Module, self).__init__()
        self.args=args
        self.dropout = nn.Dropout(args.dropout)
        self.activation=nn.GELU()
        self.weight_sum=nn.Parameter(torch.randn(args.M,1,device='cuda'),requires_grad=True)

        # Q,K,V project
        self.conv_q=nn.Conv3d(args.num_features,args.d_model,kernel_size=1)
        self.conv_k = nn.Conv3d(args.num_features, args.d_model, kernel_size=1)
        self.conv_v = nn.Conv3d(args.num_features, args.d_model, kernel_size=1)

        # Feed Forward
        self.conv1 = nn.Conv2d(in_channels=args.d_model, out_channels=args.d_model, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=args.d_model, out_channels=args.d_model, kernel_size=1)
        # Attention norm
        self.norm1 = nn.BatchNorm2d(args.d_model,affine=True)
        self.norm2 = nn.BatchNorm2d(args.d_model,affine=True)
        # Decoder norm
        self.norm3 = torch.nn.BatchNorm2d(args.d_model,affine=True)
        # Decoder projection
        self.fc_out = nn.Conv2d(args.d_model, args.d_model,kernel_size=1)


    def forward(self, x, x_date, y_date,top_value,**kwargs):
        x_date = self.conv_q(x_date)
        y_date = self.conv_k(y_date)
        x = self.conv_v(x)
        B, D, N, M, L = x_date.shape
        # Attention
        scale = 1. / math.sqrt(D)
        scores = torch.einsum("bdnml,bdnmo->bnmlo", x_date, y_date)  # (B,N,M,L,O)
        A = torch.softmax(scale * scores, dim=-2)  # For L × O Softmax on L
        V = torch.einsum("bcnml,bnmlo->bcnmo", x, A)  # (B,C,N,M,O)
        ones = torch.ones(N, 1).to(top_value.device)
        top_value = torch.cat([ones, top_value], dim=-1)  # [N,M] Denote each according to the relevance of the Adj node as the weight of sum
        weight_sum = torch.einsum('nm,mq->nmq', top_value, self.weight_sum.clone())  # (N,M,1) For each node get different summation weights
        O = torch.einsum('bcnmo,nmq->bcnqo', V.clone(), weight_sum.clone()).squeeze(-2)  # (B,C,N,O)

        # Attention norm
        y = x = self.norm1(O)
        # Feed forward
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y))
        # ResNet+Attention norm
        y = self.norm2((x.clone() + y))
        y = self.norm3(self.fc_out(y.clone()))
        return y, A  # （B,C,N,L）



class Date2Vec(nn.Module):
    def __init__(self, num_nodes,in_features,date_feature, D2V_outputmode, output_feature,seq_len,d_mark,**kwargs):
        super(Date2Vec, self).__init__()
        args=kwargs.get('args')
        self.seq_len=seq_len
        self.num_nodes=num_nodes
        self.D2V_outputmode = D2V_outputmode
        # trend item
        self.w0 = nn.Parameter(torch.randn((1),num_nodes,args.M,seq_len, 1),requires_grad=True)
        self.b0 = nn.Parameter(torch.randn((1),num_nodes,args.M,1, 1),requires_grad=True)
        # seasonal items
        self.w = nn.Parameter(torch.randn((1),num_nodes,args.M,seq_len, D2V_outputmode - 1),requires_grad=True)
        nn.init.uniform_(self.w, 0, 2 * math.pi)
        self.b = nn.Parameter(torch.randn((1),num_nodes,args.M, 1, D2V_outputmode - 1),requires_grad=True)

        self.conv_out=nn.Conv3d((1)*D2V_outputmode,in_features,kernel_size=1,stride=1)

        # The mapping matrix A obtained for center node and neighbor node should be different. By default the first one is the weight of the center node and the rest are the weights of the neighbour nodes
        self.weight = nn.Parameter(torch.randn((in_features+date_feature, 1,args.M)),requires_grad=True)


    def forward(self, data, tau):
        B, C, N, M, L = data.shape
        mark=tau.shape[1]
        # There should be different mappings (center and neighbor) for different nodes.
        # Because theoretically the CENTER node should have the largest contribution and the NEIGHBOR node should have a lower contribution compared to the CENTER node.
        date_x=tau.unsqueeze(-2).repeat(1,1,N,M,1)[...,:L] # 把日期周期也纳入
        data=torch.cat([data,date_x],dim=1) # 在特征维度上进行concat
        data=torch.einsum('bcnml,cdm->bdnml',data,self.weight)
        tau = tau.unsqueeze(1).unsqueeze(-2) # (B,1,d_mark,1,1,L+O)
        tau= tau.repeat(1,1, 1, self.num_nodes,M, 1)  # (B,C+d_mark,d_mark,N,M,L+O)

        # Get the corresponding w (this is to get the angular frequency of the trend and seasonal terms)
        w_trend = torch.einsum('bcnml,cnmlk->bcnmk', data, self.w0).unsqueeze(-2)  # (B,C,N,m,1,1)
        w_season = torch.einsum('bcnml,cnmlk->bcnmk', data, self.w).unsqueeze(-2)  # (B,C,N,m,1,K)

        w_trend = w_trend.unsqueeze(2).repeat(1, 1, mark, 1,1, 1, 1)  # (B,C,mark,N,m,1,1)
        w_season = w_season.unsqueeze(2).repeat(1, 1, mark,1, 1, 1, 1)  # (B,C,mark,N,m,1,k)
        tau = tau.unsqueeze(-1)
        b0 = self.b0.unsqueeze(1).repeat(1, mark,1, 1, 1, 1)
        b = self.b.unsqueeze(1).repeat(1, mark,1, 1, 1, 1)
        v2 = torch.matmul(tau, w_trend) + b0
        v1 = torch.matmul(tau, w_season) + b
        # Create an indexed tensor to determine whether to use sin or cos
        indices = torch.arange(v1.shape[-1], device=v1.device)
        sin_part = torch.sin(v1[..., indices % 2 == 0])
        cos_part = torch.cos(v1[..., indices % 2 == 1])
        v1_final = torch.zeros_like(v1)
        v1_final[..., indices % 2 == 0] = sin_part
        v1_final[..., indices % 2 == 1] = cos_part
        v1 = torch.mean(v1_final,dim=2)
        v2 = torch.mean(v2,dim=2)
        out = torch.cat([v2, v1], -1)  # (B,C,N,M,L,D2V_mode)
        B, C, N, M, L, K = out.shape
        out=out.permute(0,1,-1,2,3,4)
        out = out.reshape(B, -1, N,M, L)  # (B,C*D2V_mode,M,L)
        output = self.conv_out(out)

        return output

class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim,  out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """

        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))      # MLP
        hidden = hidden + input_data                           # residual
        return hidden

class GLU(nn.Module):
    def __init__(self, in_features,out_features, dropout=0.1):
        super(GLU, self).__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, (1, 1))
        self.conv2 = nn.Conv2d(in_features, in_features, (1, 1))
        self.conv3 = nn.Conv2d(in_features, in_features, (1, 1))
        self.conv_out = nn.Conv2d(in_features, out_features, (1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x,**kwargs):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        out = x1 * torch.sigmoid(x2)
        out = self.dropout(out)
        out = self.conv3(out)
        out=self.conv_out(out)
        return out
