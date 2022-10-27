import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

class PositionalEncoding(nn.Module):
    
    def __init__(self, d_hid, max_seq_length=200):
        ''' Init the sinusoid position encoding table '''
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(d_hid,max_seq_length))

    def _get_sinusoid_encoding_table(self, d_hid,max_seq_length):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_i // 2) / d_hid) for hid_i in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(max_seq_length)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature):
        super(ScaledDotProductAttention,self).__init__()
        self.temperature = temperature

    def forward(self, q, k, v):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)

        return output

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module. Allows the model to jointly attend to information
    from different representation subspaces as described in the paper:
    `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_. '''

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention,self).__init__()
        
        assert d_model % n_head == 0, 'must be correct size.'
        
        # The dimention of key and value
        d_k = d_v = d_model // n_head
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        q = self.attention(q, k, v)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.fc(q)
        q += residual

        q = self.layer_norm(q)

        return q
    
class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid):
        super(PositionwiseFeedForward,self).__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x += residual

        x = self.layer_norm(x)

        return x
    
class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model,n_head)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner)

    def forward(self, enc_input):
        enc_output = self.self_attn(enc_input, enc_input, enc_input)
        
        enc_output = self.pos_ffn(enc_output)
        return enc_output
        
class AlgoTransformer(nn.Module):
    ''' A transformer encoder model with self attention mechanism. '''

    def __init__(self, d_model=512, d_inner=2048,n_layers=6, n_head=8,enable_pe=True, max_seq_length=200):

        super(AlgoTransformer,self).__init__()
        self.layer_stack = nn.ModuleList([EncoderLayer(d_model, d_inner, n_head) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        
        if enable_pe:
            self.position_enc = PositionalEncoding(d_model, max_seq_length)
        else:
            self.position_enc = None
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, seq):

        if self.position_enc is not None:
            enc_output = self.position_enc(seq)
            enc_output = self.layer_norm(enc_output)
        else:
            enc_output = self.layer_norm(seq)

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output)

        return enc_output.reshape(enc_output.size(0),-1)
                
if __name__ == "__main__":
    
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() and True else 'cpu')
    D_MODEL = 512
    D_INNER = 1024
    
    def export_model(train_net:nn.Module):
        #https://docs.microsoft.com/zh-cn/windows/ai/windows-ml/tutorials/pytorch-analysis-convert-model
        # Setting eval model 
        train_net.eval()
        
        seq = torch.rand((1,16,D_MODEL))        
        # Input shape
        dummy_input = (seq)
        # the model's input names
        input_names = ['src']
        # the model's output names, output 8 heads
        output_names = ['outs']
        torch.onnx.export(train_net, dummy_input, "1.onnx", input_names = input_names, output_names = output_names)
        
    train_net = AlgoTransformer(d_model=D_MODEL, d_inner=D_INNER,n_layers=2, n_head=16)
    #export_model(train_net)
    
    train_net.to(DEVICE)
    
    parameters = sum([np.prod(p.shape) for p in train_net.parameters()])
    print('parameters number is:',str(parameters))
    
    for _ in range(4):
        # The length of batch ,sequence , vector 
        seq = torch.rand((1,16,D_MODEL)).to(DEVICE)
        
        start_time = time.time()
        out = train_net(seq)
        end_time = time.time()-start_time
        print('forward_time:',str(end_time))
    
    pass