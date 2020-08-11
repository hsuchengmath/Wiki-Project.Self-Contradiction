## Evaluation

#accuracy
# PA
# TP(PP)
# FP(PN)
# FN(NP)    
# TN(NN)

def Evaluation(Y_hat,Y):
    from sklearn.metrics import f1_score
    TP,FP,FN,TN = 0,0,0,0
    for i in range(len(Y_hat)):
        if int(Y_hat[i]) == 1 and int(Y[i]) ==1:
            TP+=1
        elif int(Y_hat[i]) == 1 and int(Y[i]) ==0:
            FP +=1
        elif int(Y_hat[i]) == 0 and int(Y[i]) ==1:
            FN +=1
        elif int(Y_hat[i]) == 0 and int(Y[i]) ==0:
            TN +=1
        else:
            print('[ERROR]')
    Accuracy = (TP+TN)/(TP+FP+FN+TN)
    Precision = (TP)/(TP+FP)
    Recall = (TP)/(TP+FN)
    F1 = f1_score(Y, Y_hat)

    print('Accuracy:',Accuracy)
    print('Precision:',Precision)
    print('Recall:',Recall)
    print('F1:',F1)


##doc2vec
import pyprind
#from gensim.models.doc2vec import Doc2Vec, TaggedDocument
def Doc2Vec_Layer(X):
    documents_tag,cumulate_num = list(),0
    for i in pyprind.prog_bar(range(len(X))):
        X_i = X[i]
        for j in range(len(X_i)):
            documents_tag.append(TaggedDocument(X_i[j],[cumulate_num]))
            cumulate_num +=1
    documents_tag.append(TaggedDocument(['Empty@@'],[cumulate_num]))#
    doc2vec_model = Doc2Vec(dm=1, vector_size=8, window=5, negative=5, hs=0, min_count=2, workers=4)
    doc2vec_model.build_vocab(documents_tag)
    doc2vec_model.train(documents_tag, total_examples=doc2vec_model.corpus_count, epochs=10)

    max_sentenceNUM_length = max([len(X[i]) for i in range(len(X))])#
    X_vec = list()
    for i in pyprind.prog_bar(range(len(X))):
        X_i,X_i_vec = X[i],list()
        for j in range(len(X_i)):
            X_i_vec.append(doc2vec_model.infer_vector(X_i[j]))
        for j in range(max_sentenceNUM_length-len(X_i)):#
            X_i_vec.append(doc2vec_model.infer_vector(['Empty@@']))#
        X_vec.append(X_i_vec)
    return X_vec


##word2vec
import numpy as np
#from gensim.models import Word2Vec
def Word2Vec_Layer(X):
    documents_tag = list()
    for i in range(len(X)):
        X_i = X[i]
        for j in range(len(X_i)):
            documents_tag.append(X_i[j])
    #documents_tag.append(['Empty@@'])
    word2vec_model = Word2Vec(documents_tag, min_count=1,size= 32,workers=3, window =3, sg = 1)

    X_vec = list()
    for i in range(len(X)):
        X_i,X_i_vec = X[i],list()
        for j in range(len(X_i)):
            X_i_j_vec = np.concatenate(word2vec_model[X_i[j][k]].reshape(1,-1) for k in range(len(X_i[j])))
            X_i_vec.append(X_i_j_vec)
        X_vec.append(X_i_vec)
    return X_vec



## LSTM Layer
import torch.nn as nn
class LSTM_model(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTM_model,self).__init__()
        self.input_dim,self.hidden_dim = input_dim,hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.attention_weight = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid_layer = nn.Sigmoid()
        
    def Attention_Layer(self,hidden_vec):
        self.attn = self.attention_weight(hidden_vec)
        out = torch.sum(hidden_vec * self.attn,1)
        return out

    def forward(self, X):
        hidden_vec, (h_n, c_n) = self.lstm(X)
        attn_layer_out = self.Attention_Layer(hidden_vec)
        out = self.fc(attn_layer_out)
        out = self.sigmoid_layer(out)
        return out

## Self-Attention Layer
import torch
import torch.nn as nn
from torch.autograd import Variable
import math

class Self_Attention_Layer(nn.Module):
    def __init__(self,input_dim,hidden_dim,muti_head_num):
        super(Self_Attention_Layer,self).__init__()
        self.muti_head_num = muti_head_num
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.softmax = nn.Softmax(dim=-1)

        self.WQ_head_list = [Variable(torch.zeros(input_dim,hidden_dim).type(torch.FloatTensor),requires_grad=True) for i in range(muti_head_num)]
        self.WQ_head_list = [torch.nn.init.xavier_uniform_(WQ_i) for WQ_i in self.WQ_head_list]
        self.WK_head_list = [Variable(torch.zeros(input_dim,hidden_dim).type(torch.FloatTensor),requires_grad=True) for i in range(muti_head_num)]
        self.WK_head_list = [torch.nn.init.xavier_uniform_(WK_i) for WK_i in self.WK_head_list]
        self.WV_head_list = [Variable(torch.zeros(input_dim,hidden_dim).type(torch.FloatTensor),requires_grad=True) for i in range(muti_head_num)]
        self.WV_head_list = [torch.nn.init.xavier_uniform_(WV_i) for WV_i in self.WV_head_list]

        ##classified-layer
        self.attention_weight = nn.Linear(muti_head_num*hidden_dim, 1)
        self.fc = nn.Linear(muti_head_num*hidden_dim, 1)
        self.sigmoid_layer = nn.Sigmoid()

    def forward(self,X):
        '''
        X: (bz,l,dim)
        output: (bz,l,dim*head_num)
        '''
        WQ_X_head_list = list()
        for WQ_i in self.WQ_head_list:
            WQ_i_X = torch.matmul(X, WQ_i.unsqueeze(0))
            WQ_X_head_list.append(WQ_i_X)
        WQ_X = torch.cat(WQ_X_head_list,2)

        WK_X_head_list = list()
        for WK_i in self.WK_head_list:
            WK_i_X = torch.matmul(X, WK_i.unsqueeze(0))
            WK_X_head_list.append(WK_i_X)
        WK_X = torch.cat(WK_X_head_list,2)

        WV_X_head_list = list()
        for WV_i in self.WV_head_list:
            WV_i_X = torch.matmul(X, WV_i.unsqueeze(0))
            WV_X_head_list.append(WV_i_X)
        WV_X = torch.cat(WV_X_head_list,2)

        self.attention_score = self.softmax(torch.matmul(WQ_X,WK_X.transpose(-2,-1))/math.sqrt(self.muti_head_num * self.hidden_dim)) 
        out = torch.matmul(self.attention_score,WV_X)

        out = self.Classified_Layer(out)
        return out

    def Classified_Layer(self,hidden_X):
        self.attn = self.attention_weight(hidden_X)
        attn_layer_out = torch.sum(hidden_X * self.attn,1)
        out = self.sigmoid_layer(self.fc(attn_layer_out))
        return out

# X =  torch.randn(32,9,8)
# self_attention_layer = Self_Attention_Layer(input_dim=8,hidden_dim=8,muti_head_num=3)
# self_attention_layer(X)



## retagnn-layer + self_attention-layer
from RetaGNN_layer import RAGCNConv
#from heatmap import HeatMap
class RetaGNN_SA_Model(nn.Module):
    def __init__(self,input_dim,hidden_dim,node_num):
        super(RetaGNN_SA_Model,self).__init__()
        muti_head_num = 3
        num_relations = 5

        self.node_embeddings = nn.Embedding(node_num, input_dim, padding_idx=0)
        self.node_embeddings.weight.data.normal_(0, 1.0 / self.node_embeddings.embedding_dim)
        self.conv = RAGCNConv(in_channels=input_dim,out_channels=input_dim,num_relations=num_relations, num_bases=4)
    
        ##self_attention-layer
        self.muti_head_num = muti_head_num
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.softmax = nn.Softmax(dim=-1)
        self.WQ_head_list = [Variable(torch.zeros(input_dim,hidden_dim).type(torch.FloatTensor),requires_grad=True) for i in range(muti_head_num)]
        self.WQ_head_list = [torch.nn.init.xavier_uniform_(WQ_i) for WQ_i in self.WQ_head_list]
        self.WK_head_list = [Variable(torch.zeros(input_dim,hidden_dim).type(torch.FloatTensor),requires_grad=True) for i in range(muti_head_num)]
        self.WK_head_list = [torch.nn.init.xavier_uniform_(WK_i) for WK_i in self.WK_head_list]
        self.WV_head_list = [Variable(torch.zeros(input_dim,hidden_dim).type(torch.FloatTensor),requires_grad=True) for i in range(muti_head_num)]
        self.WV_head_list = [torch.nn.init.xavier_uniform_(WV_i) for WV_i in self.WV_head_list]

        ##classified-layer
        self.attention_weight = nn.Linear(muti_head_num*hidden_dim, 1)
        self.fc = nn.Linear(muti_head_num*hidden_dim, 1)
        self.sigmoid_layer = nn.Sigmoid()

    def Self_Attention_Layer(self,X):
        '''
        X: (bz,l,dim)
        output: (bz,l,dim*head_num)
        '''
        WQ_X_head_list = list()
        for WQ_i in self.WQ_head_list:
            WQ_i_X = torch.matmul(X, WQ_i.unsqueeze(0))
            WQ_X_head_list.append(WQ_i_X)
        WQ_X = torch.cat(WQ_X_head_list,2)

        WK_X_head_list = list()
        for WK_i in self.WK_head_list:
            WK_i_X = torch.matmul(X, WK_i.unsqueeze(0))
            WK_X_head_list.append(WK_i_X)
        WK_X = torch.cat(WK_X_head_list,2)

        WV_X_head_list = list()
        for WV_i in self.WV_head_list:
            WV_i_X = torch.matmul(X, WV_i.unsqueeze(0))
            WV_X_head_list.append(WV_i_X)
        WV_X = torch.cat(WV_X_head_list,2)

        self.attention_score = self.softmax(torch.matmul(WQ_X,WK_X.transpose(-2,-1))/math.sqrt(self.muti_head_num * self.hidden_dim)) 
        #print('attention_score:',self.attention_score.shape)
        out = torch.matmul(self.attention_score,WV_X)
        return out

    def Classified_Layer(self,hidden_X):
        self.attn = self.attention_weight(hidden_X)
        attn_layer_out = torch.sum(hidden_X * self.attn,1)
        out = self.sigmoid_layer(self.fc(attn_layer_out))
        return out

    def forward(self,batch_X,SA_visualization=False):
        X = list()
        for i in range(len(batch_X)):
            sent_ids = batch_X[i][0]
            edge_index = batch_X[i][1]
            edge_type = batch_X[i][2]
            node_ids = batch_X[i][3]
            node_ids = torch.tensor(node_ids)
            edge_index = torch.tensor(edge_index).transpose(1,0)
            edge_type = torch.tensor(edge_type)
            x = self.node_embeddings(node_ids)
            x_updated = self.conv(x,edge_index,edge_type)
            sent_embedding = x_updated[sent_ids].view(1,-1,self.input_dim)
            X.append(sent_embedding)
        X = torch.cat(X,0)
        hidden_X = self.Self_Attention_Layer(X)
        out = self.Classified_Layer(hidden_X)
        if SA_visualization is True:
            # for i in range(self.attention_score.size()[0]):
            #     attention_score = self.attention_score[i,:,:].cpu().data.numpy()
            #     HeatMap(attention_score,i)
            attention_score_numpy = self.attention_score.cpu().data.numpy()
            out_numpy = out.cpu().data.numpy()
            return out_numpy,attention_score_numpy
        else:
            return out

  


## train process layer

import torch
import torch.nn as nn
import torch.optim as optim
#train_X = [(b,l,d),(b,l,d),...] ; train_Y = [(b,),(b,),...]
#test_X = (N,l,d)  test_Y = (N,)

def Train_Eval_Process_Layer(train_X,train_Y,test_X,test_Y):
    # LSTM
    epoch_num = 25
    #model = LSTM_model(input_dim=8,hidden_dim=8)
    model = Self_Attention_Layer(input_dim=8,hidden_dim=8,muti_head_num=3)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCELoss()
    for epoch_  in range(epoch_num):
        model.train()
        for i in range(len(train_X)):
            batch_X,batch_Y = train_X[i],train_Y[i] #(b,l,d) ,(b,)
            batch_Y_hat = model(batch_X)
            batch_Y_hat = batch_Y_hat.squeeze(dim=-1)
            loss = criterion(batch_Y_hat, batch_Y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print('loss:',loss)
        model.eval()
        test_Y_hat = model(test_X).cpu().data.numpy()
        test_Y_hat_list = list()
        for i in range(test_Y_hat.shape[0]):
            if test_Y_hat[i,0] >= 0.5:
                test_Y_hat_list.append(1)
            else:
                test_Y_hat_list.append(0)
        Evaluation(test_Y_hat_list,test_Y)
        
    
#single2batch layer for LSTM 
import numpy as np
def Single2Batch_Layer(X,Y):
    train_rate = 0.8
    batch_size = 32
    input_dim = 8
    train_size = int(train_rate * len(X))
    import random
    data_index = [i for i in range(len(X))]
    shuffle_data_index = random.sample(data_index,len(data_index))
    train_data_index = shuffle_data_index[:train_size]
    test_data_index = shuffle_data_index[train_size:]
    if train_size % batch_size != 0 :
        batch_num = int(train_size / batch_size) + 1 
    else:
        batch_num = int(train_size / batch_size) 
    train_X,train_Y = list(),list()
    X_virtual,Y_virtual = list(),list()
    for i in range(len(train_data_index)):
        X_i,Y_i = X[train_data_index[i]],Y[train_data_index[i]]
        Y_i = np.array([Y_i])
        X_virtual.append(X_i.reshape(1,-1,input_dim))
        Y_virtual.append(Y_i.reshape(1,1))
        if len(X_virtual) >= batch_size :
            X_virtual,Y_virtual = np.concatenate(X_virtual), np.concatenate(Y_virtual)
            X_virtual,Y_virtual = torch.tensor(X_virtual),torch.tensor(Y_virtual)
            train_X.append(X_virtual)
            train_Y.append(Y_virtual)
            X_virtual,Y_virtual = list(),list()
        elif len(train_X) == batch_num - 1 and i == len(train_data_index)-1:
            X_virtual,Y_virtual = np.concatenate(X_virtual), np.concatenate(Y_virtual)
            X_virtual,Y_virtual = torch.tensor(X_virtual),torch.tensor(Y_virtual)
            train_X.append(X_virtual)
            train_Y.append(Y_virtual)
    test_X,test_Y = list(),list()
    for i in range(len(test_data_index)):
        X_i,Y_i = X[test_data_index[i]],Y[test_data_index[i]]
        X_i = X_i.reshape(1,-1,input_dim)
        test_X.append(X_i)
        test_Y.append(Y_i)
    test_X = np.concatenate(test_X)
    test_X = torch.tensor(test_X)
    return train_X,train_Y,test_X,test_Y





def Find_Pair_Sentence_BY_Attention_Score(test_X_text,attention_score_numpy,test_Y,test_Y_hat_list):
    for i in range(len(test_Y_hat_list)):
        if test_Y_hat_list[i] == 1 and test_Y[i] == 1:
            attn_score_np = attention_score_numpy[i,:,:]
            HeatMap(attn_score_np,i)
            print(i)
            print(np.where(attn_score_np==np.max(attn_score_np)))
            print('==============')


def Find_High_Score_Pair_Sent(test_Y_hat_list,test_Y,attention_score_numpy,test_X_text,top=10):
    # Y = 1; Y^ = 1
    # Y = 0; Y^ = 1
    selected_high_score_data = list()
    for i in range(len(test_Y)):
        true_Y,pred_Y = test_Y[i],test_Y_hat_list[i]
        if true_Y == 1 and pred_Y == 1 or true_Y == 0 and pred_Y == 1:
            high_score_data = list()
            sent_num = len(test_X_text[i])
            attention_score_i = attention_score_numpy[i,:sent_num-1,:sent_num-1]
            attention_score_i_flat = list(attention_score_i.flat)
            attention_score_i_flat_top = sorted(attention_score_i_flat,reverse=True)[:top]
            top_pair_sent = list()
            for j in range(len(attention_score_i_flat_top)):
                value_ = attention_score_i_flat_top[j]
                coordinate_ = np.where(attention_score_i == value_)
                X_coordinate_,Y_coordinate_ = coordinate_[0][0],coordinate_[1][0]
                X_sent_,Y_sent_ = test_X_text[i][X_coordinate_],test_X_text[i][Y_coordinate_]
                top_pair_sent.append([X_sent_,Y_sent_])
            high_score_data.append(test_X_text[i])
            high_score_data.append(top_pair_sent)
            high_score_data.append(attention_score_i)
            high_score_data.append([true_Y,pred_Y])
            selected_high_score_data.append(high_score_data)
    #print(selected_high_score_data)
    return selected_high_score_data
        




def Train_Eval_Process_Layer_v2(train_X,train_Y,test_X,test_Y,node_num,test_X_text):
    # RetaGNN + Self Attention
    import pyprind
    import pickle
    epoch_num =25
    input_dim = 8
    hidden_dim = 8
    model = RetaGNN_SA_Model(input_dim,hidden_dim,node_num)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCELoss()
    for epoch_  in range(epoch_num):
        model.train() 
        for i in pyprind.prog_bar(range(len(train_X))):
            batch_X,batch_Y = train_X[i],train_Y[i] #(b,l,d) ,(b,)
            batch_Y_hat = model(batch_X).squeeze(dim=-1)
            loss = criterion(batch_Y_hat, batch_Y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print('loss:',loss)
        model.eval()
        if epoch_ != epoch_num-1:
            test_Y_hat = model(test_X,SA_visualization=False).cpu().data.numpy()
            #test_Y_hat,attention_score_numpy = model(test_X,SA_visualization=True)
        else:
            test_Y_hat,attention_score_numpy = model(test_X,SA_visualization=True)
        test_Y_hat_list = list()
        for i in range(test_Y_hat.shape[0]):
            if test_Y_hat[i,0] >= 0.5:
                test_Y_hat_list.append(1)
            else:
                test_Y_hat_list.append(0)
        Evaluation(test_Y_hat_list,test_Y)
        if epoch_ == epoch_num-1:     
            selected_high_score_data = Find_High_Score_Pair_Sent(test_Y_hat_list,test_Y,attention_score_numpy,test_X_text)
            a_dict = {'selected_high_score_data':selected_high_score_data}
            ##pickle a va riable to a file
            file = open('stock/selected_high_score_data.pickle', 'wb')
            pickle.dump(a_dict, file)
            file.close()











#single2batch layer for RetaGNN + Self Attention 
import numpy as np
def Single2Batch_Layer_v2(X,Y,X_text):
    train_rate = 0.8
    batch_size = 32
    train_size = int(train_rate * len(X))
    import random
    data_index = [i for i in range(len(X))]
    shuffle_data_index = random.sample(data_index,len(data_index))
    train_data_index = shuffle_data_index[:train_size]
    test_data_index = shuffle_data_index[train_size:]
    if train_size % batch_size != 0 :
        batch_num = int(train_size / batch_size) + 1 
    else:
        batch_num = int(train_size / batch_size) 
    train_X,train_Y = list(),list()
    X_virtual,Y_virtual = list(),list()
    for i in range(len(train_data_index)):
        X_i,Y_i = X[train_data_index[i]],Y[train_data_index[i]]        
        Y_i = np.array([Y_i])
        X_virtual.append(X_i)
        Y_virtual.append(Y_i.reshape(1,1))
        if len(X_virtual) >= batch_size :
            Y_virtual = np.concatenate(Y_virtual)
            Y_virtual = torch.tensor(Y_virtual)
            train_X.append(X_virtual)
            train_Y.append(Y_virtual)
            X_virtual,Y_virtual = list(),list()
        elif len(train_X) == batch_num - 1 and i == len(train_data_index)-1:
            Y_virtual =  np.concatenate(Y_virtual)
            Y_virtual = torch.tensor(Y_virtual)
            train_X.append(X_virtual)
            train_Y.append(Y_virtual)
    test_X,test_Y,test_X_text = list(),list(),list()
    for i in range(len(test_data_index)):
        X_i,Y_i = X[test_data_index[i]],Y[test_data_index[i]]
        X_i_text = X_text[test_data_index[i]]
        test_X.append(X_i)
        test_Y.append(Y_i)
        test_X_text.append(X_i_text)
    return train_X,train_Y,test_X,test_Y,test_X_text
