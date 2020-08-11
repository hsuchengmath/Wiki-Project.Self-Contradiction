import json
import pickle
from test_graph import Sent2Word2Word_Construct_Graph,Word2Entity2Type,Construct_Graph



with open('stock/self_contradictory_XY_2_v2.pickle', 'rb') as file:
    self_contradictory_XY_2 =pickle.load(file)

X = self_contradictory_XY_2['X']
Y = self_contradictory_XY_2['Y']
title = self_contradictory_XY_2['title']
graph_info = Sent2Word2Word_Construct_Graph(X)
new_graph_info = Word2Entity2Type(X,graph_info)

 

import os 
if os.path.exists('stock/self_contradictory_XY_3_v2.json'):
    fileTest = 'stock/self_contradictory_XY_3_v2.json'
    try:
        os.remove(fileTest)
    except OSError as e:
        print(e)
    else:
        print("File is deleted successfully")

with open('stock/self_contradictory_XY_3_v2.json','a') as outfile:
    json.dump(a_dict,outfile,ensure_ascii=False)
    outfile.write('\n')
