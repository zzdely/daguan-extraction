'''
Created on 2019年7月3日

@author: guolixiang
'''
from keras.preprocessing.sequence import pad_sequences
import numpy as np

def get_sentences(fpath):
    with open(fpath, "r", encoding="utf-8") as f:
        lines = f.readlines()
        sents = []
        bio_labels = []
        for line in lines:
            sent = []
            bio_label = []
            line = line.strip()
            span_list = line.split("  ") # 每一个标记的span
            for span in span_list:
                chars_list = span.split("/")[0].split("_") # 字的序列
                label = span.split("/")[1] # 标记
                sent.extend(chars_list)
                
                if label == "o":
                    t = ["o"]*len(chars_list)
                else:
                    t = ["B-"+label]
                    t_I = ["I-"+label]*(len(chars_list)-1)
                    t.extend(t_I)
                bio_label.extend(t)
            sents.append(sent)
            bio_labels.append(bio_label)
    
    return sents, bio_labels # 返回句子序列和BIO序列

def get_test_sentences(fpath):
    with open(fpath, "r", encoding="utf-8") as f:
        lines = f.readlines()
        sents = []
        for line in lines:
            line = line.strip()
            sents.append(line.split("_"))
    
    return sents # 返回测试集的句子序列
    
    
    

def build_bio_label_list(role_label):
    '''
    根据role_label中包含的要素，生成BIO标记集合，还要加上<PAD>标记
    '''
    bio_label_list = ["o", "<PAD>"]
    for s in role_label:
        bio_label_list.append("B-"+s)
        bio_label_list.append("I-"+s)
    
    # bio_label_list: ["O", "<PAD>", "B-trigger", "I-trigger", ...]
    return bio_label_list

def token2index(token_list):
    '''
    将token_list （一般是词典序列或者标签序列）转化为id，并返回token2id和id2token的dict
    '''
    token2id = {}
    id2token = {}
    for i, token in enumerate(token_list):
        id2token[i] = token
        token2id[token] = i
    return token2id, id2token

def get_valid_label_index(label2index, invalid_labels):
    '''
    把有效的label_index作为一个list返回
    label2index: dict类型
    invalid_labels: list类型，里面每个元素是string
    '''
    valid_label_index_list = []
    for label in label2index:
        if label in invalid_labels:
            continue
        else:
            valid_label_index_list.append(label2index[label])
    
    return valid_label_index_list

def to_index_BIO(sents, label_seqs, word2index, label2index):
    '''
    同上面to_index的功能，不同的是，这里的是BIO标注模式，把BIO转化为index序列
    '''
    index_sents = []
    index_label_seqs = []
    for i,_ in enumerate(sents):
        sent_seq = [word2index[w] if w in word2index else word2index["_ukn_"] for w in sents[i]]
        label_seq = [label2index[s] for s in label_seqs[i]]
        
        index_sents.append(sent_seq)
        index_label_seqs.append(label_seq)
    
    return index_sents, index_label_seqs

def to_index_BIO_test(sents, word2index):
    '''
    同上面to_index的功能，不同的是，这里的是BIO标注模式，把BIO转化为index序列
    '''
    index_sents = []
    for i,_ in enumerate(sents):
        sent_seq = [word2index[w] if w in word2index else word2index["_ukn_"] for w in sents[i]]
        index_sents.append(sent_seq)
 
    return index_sents

def padding_BIO(index_sents, index_label_seqs, maxlen, word2index, label2index):
    padded_index_sents = pad_sequences(index_sents,value=word2index["_pad_"],padding='post',maxlen=maxlen)
    padded_index_label_seqs = pad_sequences(index_label_seqs,value=label2index["<PAD>"],padding='post',maxlen=maxlen)
    return padded_index_sents, padded_index_label_seqs

def padding_BIO_test(index_sents, maxlen, word2index):
    padded_index_sents = pad_sequences(index_sents,value=word2index["_pad_"],padding='post',maxlen=maxlen)
    return padded_index_sents

def get_sents_length(sents):
    '''
    记录测试集中每个句子的真实长度
    '''
    return [len(s) for s in sents]

def trans_index2label(index_tag_list, index2label):
    all_label_list = []
    for sent_tag in index_tag_list:
        all_label_list.append([index2label[t] if index2label[t] != "<PAD>" else "o" for t in sent_tag])
    
    return all_label_list

def form(sents, all_label_list):
    str_all_sents_with_label = []
    for i,label_list in enumerate(all_label_list):
        
        sent_with_label = ""
        cursor = 0 # 游标，指示当前遍历到bio标记序列的哪个地方
        while cursor < len(label_list):
            if label_list[cursor].startswith("B-") or label_list[cursor].startswith("I-"):
                current_role_label = label_list[cursor].replace("B-", "").replace("I-", "") # 获取当前角色标记，可能是trigger，time, location, participant
                start_index = cursor
                end_index = cursor
                
                cursor = cursor + 1
                while cursor < len(label_list):
                    if label_list[cursor].replace("B-", "").replace("I-", "") == current_role_label:
                        end_index = cursor
                        cursor = cursor + 1
                    else:
                        break
                        
#                 span = (start_index, end_index)
                sent_with_label = sent_with_label+ "_".join(sents[i][start_index:end_index+1]) + "/" + current_role_label + "  "
#                 span_label = [span, current_role_label]
#                 span_label_list.append(span_label)
            elif label_list[cursor] == "o":
                current_role_label = label_list[cursor]
                start_index = cursor
                end_index = cursor
                
                cursor = cursor + 1
                while cursor < len(label_list):
                    if label_list[cursor] == "o":
                        end_index = cursor
                        cursor = cursor + 1
                    else:
                        break
                
                sent_with_label = sent_with_label+ "_".join(sents[i][start_index:end_index+1]) + "/" + current_role_label + "  "
            elif label_list[cursor] == "<PAD>":
                current_role_label = label_list[cursor]
                start_index = cursor
                end_index = cursor
                
                cursor = cursor + 1
                while cursor < len(label_list):
                    if label_list[cursor] == "<PAD>":
                        end_index = cursor
                        cursor = cursor + 1
                    else:
                        break
                
                sent_with_label = sent_with_label+ "_".join(sents[i][start_index:end_index+1]) + "/" + current_role_label + "  "
            else:
                cursor = cursor + 1
                
        sent_with_label = sent_with_label.strip()
        str_all_sents_with_label.append(sent_with_label)
    return str_all_sents_with_label
        
                        
                
            














                        
                
            