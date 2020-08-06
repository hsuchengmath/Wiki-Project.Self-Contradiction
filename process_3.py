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
