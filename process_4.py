













import json
import pickle
import pyprind



with open("stock/self_contradictory_XY_3_v2.json",'r') as load_f:
    a = list(load_f)[0].replace('\n','')
    self_contradictory_XY_3 = json.loads(a)


X_text = self_contradictory_XY_3['X']
Y = self_contradictory_XY_3['Y']
title = self_contradictory_XY_3['title']
graph_info = self_contradictory_XY_3['graph_info']


object2entity_ids_list,edge_index_list,edge_type_list = graph_info[0],graph_info[1],graph_info[2]
sentence_ids_list,node_no_list,node_num_give_model = graph_info[3],graph_info[4],graph_info[5]
 
sentence_num_list = [len(sentence_ids_list[i]) for i in range(len(sentence_ids_list))]
max_sentence_num  = max(sentence_num_list)
import pyprind
X = list()
for i in pyprind.prog_bar(range(len(sentence_ids_list))):
    object2entity_ids = object2entity_ids_list[i]
    tocken_ids = object2entity_ids['sentence-'+ 'tocken']
    sentence_ids = sentence_ids_list[i]
    remain_sentence_num = max_sentence_num - len(sentence_ids)
    for j in range(remain_sentence_num):
        sentence_ids.append(tocken_ids)
    edge_index = edge_index_list[i]
    edge_type = edge_type_list[i]
    node_no = node_no_list[i]
    node_ids = [k for k in range(node_no)]
    X.append([sentence_ids,edge_index,edge_type,node_ids])

from backup_tool import Single2Batch_Layer_v2,Train_Eval_Process_Layer_v2
train_X,train_Y,test_X,test_Y,test_X_text = Single2Batch_Layer_v2(X,Y,X_text)
Train_Eval_Process_Layer_v2(train_X,train_Y,test_X,test_Y,node_num_give_model,test_X_text)
