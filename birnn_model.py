'''
Created on 2019年6月22日

@author: guolixiang

baseline模型，采用BiGRU, 加上全连接层，再加上softmax
'''
import torch
from torch import nn
import torch.nn.functional as F

class BiRNN_Model(nn.Module):
    def __init__(self, maxlen, hidden_n, embeddings, bio_label_list):
        '''
        Constructor
        maxlen: 序列最大长度
        hidden_n: RNN隐藏层大小
        role_label: 要素标记空间，list类型，形如["trigger", "time", "location", "participant",...],有顺序
        span_list: 是所有可能的span组合，一共T(T+1)/2种，其中T是句子长度seq_len，形如[(0,0), (0,1),...,(T-2,T-1), (T-1,T-1)], 有顺序
        r_core: 核心要素，目前只有trigger，是一个集合对象
        '''
        super().__init__()
        self.maxlen = maxlen
        self.hidden_n = hidden_n
#         self.role_label = role_label # 要素标记空间，list类型
        
        self.span_list = [(i,j) for i in range(maxlen) for j in range(i,maxlen)]
#         self.r_core = r_core
        
        embed_dim = embeddings.shape[-1] # 词向量维数
        embeddings = torch.from_numpy(embeddings) # 把numpy类型的embeddings转化为torch类型的embeddings
        self.embeds = nn.Embedding.from_pretrained(embeddings, freeze=True) # 引入预训练词向量，freeze参数控制使得词向量在训练过程中不更新
        
        # 双向GRU
        self.bigru = nn.GRU(input_size=embed_dim, 
                        hidden_size=self.hidden_n, 
                        batch_first=True, 
                        bidirectional=True)
        
        self.linear_out_size = 100 # 加入一个全连接层，定义全连接层的输出size大小为100
        self.linear = nn.Linear(2*self.hidden_n, self.linear_out_size)
        self.bio_label_list = bio_label_list
        # 加入一个softmax层，实际上是先有一个全连接层，然后是一个softmax函数操作
        self.out = nn.Linear(self.linear_out_size, len(self.bio_label_list))
        
        # pytorch的官方文档里对nn.Softmax有如下一段说明，意思是说如果后面用NLLLoss损失函数的话，用LogSoftmax会更好，因为NLLLoss用的是log-probability
        # （后来在NLLLoss的说明中发现，如果不想用LogSoftmax的话，可以用CrossEntropyLoss损失函数，效果也是一样的）
        # This module doesn’t work directly with NLLLoss, 
        # which expects the Log to be computed between the Softmax and itself. 
        # Use LogSoftmax instead (it’s faster and has better numerical properties).
#         self.sofmax = nn.Softmax()
        
    def forward(self, x):
        # x: (batch, seq_len, embed_dim) x是输入的batch文本序列
        x = self.embeds(x).float()
        # output: (batch, seq_len, 2*hidden_size) 输出的是每个词的隐向量表示，因为是双向GRU所以是2*hidden_size
        output, _ = self.bigru(x)
        # output: (batch, seq_len, self.linear_out_size)
        output = self.linear(output)
        output = F.relu(output) # 加入激活函数relu
        # output: (batch, seq_len, len(self.bio_label_list))
        output = self.out(output)
#         output = self.sofmax(output)
        return output
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        