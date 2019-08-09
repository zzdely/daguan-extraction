'''
Created on 2019年4月13日

@author: guolixiang

构建一个baseline方法，用sequence labeling的模型，BiGRU+Dense+softmax

'''
from preprocess import get_sentences, token2index, to_index_BIO, build_bio_label_list, padding_BIO, get_valid_label_index, get_test_sentences,\
    to_index_BIO_test, padding_BIO_test, get_sents_length, trans_index2label,\
    form
import torch
from torch import nn
import time
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from idx_dataset import IDXDataset
from tqdm import tqdm

from birnn_model import BiRNN_Model
from sklearn.metrics import f1_score

########### 模型的各种参数 #################
maxlen = 500 # 序列最大长度（经过统计都不会超过120个词）
hidden_n = 100 # RNN的隐向量维度
word2vec_model_dir = "./models/"
embeddings = np.loadtxt(word2vec_model_dir+"embedding.txt") # 词向量文件
role_label = ["a", "b", "c"] # 有哪些标签
# r_core = set(["trigger"]) # 核心标签
epochs = 200
batch_size = 64
learning_rate=0.01


############################################

start = time.time()
print("start: ", start)
# 把原始的基于BIO表示的数据变成本文研究需要的span格式，并用pickle保存为二进制文件
# full_data_10000_filter.txt是把多trigger的推文滤除后剩下的数据
# data = change2span("./data/full_data_10000_filter.txt")

data_path = "./datagrand/train.txt"
test_data_path = "./datagrand/test.txt"
# 把句子和bio分离出来
sents, bio_labels = get_sentences(data_path)
test_sents = get_test_sentences(test_data_path)
# # 把BIO标记分离出来
# label_seqs = get_bio_seq(data_path)
# # 把CU、NCU分类标记分离出来
# class_labels = get_class(data_path)

with open(word2vec_model_dir+"vocab.txt", "r", encoding="utf-8") as f:
    word_list = [w.strip() for w in f.readlines()]

# 根据标签生成BIO标记列表，包括“o”和“<PAD>”
bio_label_list = build_bio_label_list(role_label)

word2index, index2word = token2index(word_list)
label2index, index2label = token2index(bio_label_list)

# 为后面sklearn计算f1score的时候用，只考虑valid label
invalid_labels = ["o", "<PAD>"]
valid_label_index_list = get_valid_label_index(label2index, invalid_labels)

# index化
index_sents, index_label_seqs = to_index_BIO(sents, bio_labels, word2index, label2index)
index_sents_test = to_index_BIO_test(test_sents, word2index)
sents_real_len = get_sents_length(index_sents_test) # 记录测试集的每一个句子的真实长度
sents_real_len = np.array(sents_real_len) # 转化为numpy格式，方便用id列表做切片

# 加入padding，对输入句子序列和标记序列都进行padding
# y_cu_label = np.array(y_cu_label) # 必须转化成numpy格式，否则下面的y_cu_label[cu_id_list]语法错误
padded_index_data_x, y_bio_label = padding_BIO(index_sents, index_label_seqs, maxlen, word2index, label2index)
padded_index_data_x_test = padding_BIO_test(index_sents_test, maxlen, word2index)

# # 把CU事件的数据单独抽出来
# cu_id_list = []
# for i, label in enumerate(y_cu_label):
#     if label == 1: # 如果是CU事件
#         cu_id_list.append(i)
#     else:
#         pass
# padded_index_data_x = padded_index_data_x[cu_id_list]
# y_bio_label = y_bio_label[cu_id_list]
# y_cu_label = y_cu_label[cu_id_list]

########################## 进入模型阶段 ##########################
birnn_model = BiRNN_Model(maxlen, hidden_n, embeddings, bio_label_list)

device = "cuda:1" if torch.cuda.is_available() else "cpu"
# device = "cpu"
birnn_model.to(device)
params = filter(lambda param: param.requires_grad, birnn_model.parameters())

# 将权重参数w和偏置参数b分离开来，因为在L2正则化中最好不要将偏置参数也正则化，否则会造成严重欠拟合
weight_p, bias_p = [],[]
for name, p in birnn_model.named_parameters():
    if 'bias' in name:
        bias_p += [p]
    else:
        weight_p += [p]

weight = torch.ones(len(label2index))
weight[label2index["o"]] = 0.3
weight[label2index["<PAD>"]] = 0.2
weight = weight.to(device)
myloss = nn.CrossEntropyLoss(weight)
# optim = torch.optim.SGD(params=params,lr=learning_rate)
optim = torch.optim.SGD([{'params': weight_p, 'weight_decay':3e-4}, {'params': bias_p, 'weight_decay':0}], lr=learning_rate, momentum=0.9)


########################## 进入训练阶段 ##########################

# 如果不做交叉验证的话，仅仅只划分训练集和测试集，取20%做测试集
# train_x, test_x, train_y_bio, test_y_bio, train_y_cu, test_y_cu = train_test_split(padded_index_data_x, y_bio_label, y_cu_label, test_size=0.2, random_state=11, shuffle=True)
# # train_data = CUDataset(train_x, train_y_span, train_y_cu)
# # test_data = CUDataset(test_x, test_y_span, test_y_cu)
# # print(train_x)
# train_size = train_x.shape[0] # 训练样本数量
# test_size = test_x.shape[0] # 测试样本数量
# train_idx_list = np.arange(train_size)
# test_idx_list = np.arange(test_size)
# train_index = IDXDataset(train_idx_list)
# test_index = IDXDataset(test_idx_list)

train_x = padded_index_data_x
train_y_bio = y_bio_label

train_size = train_x.shape[0]
train_idx_list = np.arange(train_size)
train_index = IDXDataset(train_idx_list)

test_x = padded_index_data_x_test
test_size = test_x.shape[0] # 测试样本数量
test_idx_list = np.arange(test_size)
test_index = IDXDataset(test_idx_list)

for e in range(epochs):
    birnn_model.train()
#     data_loader = DataLoader(dataset=train_data, batch_size=batch_size,shuffle=True,num_workers=0,drop_last=False)
    data_loader = DataLoader(dataset=train_index, batch_size=batch_size,shuffle=False,num_workers=0,drop_last=False)
    train_loss, train_acc = [], []
    for batch_index in data_loader:
        optim.zero_grad()
        x, y_bio = torch.from_numpy(train_x[batch_index]).long(), torch.from_numpy(train_y_bio[batch_index]).long()
        x = x.to(device)
        y_bio = y_bio.to(device)
        score_label = birnn_model(x) # 标签得分，不需要softmax，因为CrossEntropyLoss自带log softmax
#         print("score_label:", score_label.shape)
        
        # 下面两个必须view一把，才能满足CrossEntropyLoss的输入要求
        score_label = score_label.view(score_label.shape[0]*maxlen, -1) # 这里不能直接使用batch_size*maxlen，因为最后一个batch中并没有batch_size条记录
        y_bio = y_bio.view(y_bio.shape[0]*maxlen)
        loss = myloss(score_label, y_bio)
#         print(loss)
        train_loss.append(loss.data.cpu().numpy()) # 复制到cpu上，就可以用下面的np.mean了
        loss.backward()
        optim.step()
    avg_loss = np.mean(train_loss)
    print("epoch:", e, ", avg_loss:", avg_loss)
    
    print("****************start testing****************")
    data_loader = DataLoader(dataset=test_index, batch_size=batch_size,shuffle=False,num_workers=0,drop_last=False)
    birnn_model.eval()
    with torch.no_grad():
         
#         true_list = []
#         pred_list = []
        all_real_tag_list = []
        for batch_index in data_loader:
            x = torch.from_numpy(test_x[batch_index]).long()
            x = x.to(device)
            # y_bio = y_bio.to(device)
             
            # score_label: (batch, seq_len, label_size)
            score_label = birnn_model(x)
            batch_tag = torch.argmax(score_label, -1)
#             print("batch_tag:", batch_tag.shape)
#             print("y_bio:", y_bio.shape)
            batch_tag = batch_tag.data.cpu().numpy()
             
#             f1 = f1_score(y_bio.flatten(), batch_tag.flatten(), labels=valid_label_index_list, average="micro")
#             print("f1score:", f1)
            
            batch_sents_real_len = sents_real_len[batch_index]
            batch_tag_list = list(batch_tag)
            
            batch_real_tag_list = []
            for i, tag in enumerate(batch_tag_list):
                real_tag_list = batch_tag_list[i][0:batch_sents_real_len[i]]
                batch_real_tag_list.append(real_tag_list)
            
            all_real_tag_list.extend(batch_real_tag_list)
        
        all_label_list = trans_index2label(all_real_tag_list, index2label)
        assert len(test_sents) == len(all_label_list)
        print("预测完毕，开始构造输出提交格式")
        str_all_sents_with_label = form(test_sents, all_label_list)
        print("str_all_sents_with_label长度:", len(str_all_sents_with_label))
#         print(str_all_sents_with_label)
        with open("./datagrand/submit_result.txt", "w", encoding="utf-8") as f:
            for s in str_all_sents_with_label:
                f.write(s+"\n")
        print("写入submit_result.txt")
        
                
            
#             true_list.extend(list(y_bio.flatten()))
#             pred_list.extend(list(batch_tag.flatten()))
#          
#         true = np.array(true_list)
#         pred = np.array(pred_list)
#         print("true:",true.shape)
#         print("pred:",pred.shape)
#         assert true.shape == pred.shape
#         true = true.flatten()
#         pred = pred.flatten()
#         print("true:",true.shape)
#         print("pred:",pred.shape)
#         f1 = f1_score(true, pred, labels=valid_label_index_list, average="micro")
#         print("f1score:", f1)
        
            
            
            
            
            
            
            

end = time.time()
print("end: ",end)

print("end-start: ", end-start)




    
