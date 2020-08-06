
'''
import stanza
stanza.download('en')
nlp = stanza.Pipeline('en')
text = 'Barack Obama was born in Hawaii.  He was elected president in 2008.'
def Dependency_Parsing_Sent2Gph(text):
    doc,dependency_parsing = nlp(text),list()
    for sent in doc.sentences:
        for word_doc in sent.words:
            word = word_doc.text
            if word_doc.head > 0:
                word2 = sent.words[word_doc.head-1].text
            else:
                word2 = 'root'
            edge_name = word_doc.deprel
            dependency_parsing.append([word,word2,edge_name])
    return dependency_parsing
dependency_parsing = Dependency_Parsing_Sent2Gph(text)
print(dependency_parsing)
'''


'''
import stanza
nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')
text = 'Barack Obama was born in Hawaii.  He was elected president in 2008.'
def Entity_and_Type(text):
    doc = nlp(text)
    entity_and_type = list()
    for sent in doc.sentences:
        for ent in sent.ents:
            entity = ent.text
            entity_type = ent.type
            entity_and_type.append([entity,entity_type])
    return entity_and_type
'''





'''
print('====')
import json
import mwparserfromhell

path = '/home/hsucheng/wiki/dataset/' + 'solved.jsonl'

with open(path, 'r') as jsonlfile:
    jsonlfile = list(jsonlfile)



a = json.loads(jsonlfile[130])
print(a.keys())
print('===========================')
#print(a['solvepovTime'])
print('===========================')
#print(a['povVersion'])

import re
from nltk.tokenize import sent_tokenize
text = a['povVersion']
wikicode = mwparserfromhell.parse(text)
templates = wikicode.filter_templates()

str_wikicode = str(wikicode)
for j in range(len(templates)):
    str_wikicode = str_wikicode.replace(str(templates[j]),' ')
str_wikicode = str_wikicode.replace('[[','')
str_wikicode = str_wikicode.replace(']]','')
str_wikicode = sent_tokenize(str_wikicode)



import pickle

a_dict = {'str_wikicode':str_wikicode}

# pickle a variable to a file
file = open('stock/example_from_test_graph.pickle', 'wb')
pickle.dump(a_dict, file)
file.close()

'''



import pickle
import stanza
import pyprind
# reload a file to a variable
with open('stock/example_from_test_graph.pickle', 'rb') as file:
    example_from_test_graph =pickle.load(file)
str_wikicode = example_from_test_graph['str_wikicode']
 
# import pyprind
# import stanza
# graph architecture
# s1,s2,...,sN 
# w1,w2,...,wn 
# e1,e2,.... 
# ty1,ty2,....

# s-w :0,1  ; w-w :2  ; w-e :3 ; e-ty :4
# tocken sentence key=> 'sentence-'+ 'tocken'
# env : py3.6

#multi_str_wikicode = [str_wikicode for i in range(3)]

def Sent2Word2Word_Construct_Graph(multi_str_wikicode):
    object2entity_ids_list = list()
    edge_index_list,edge_type_list,index_list = list(),list(),list()
    sentence_ids_list = list()

    nlp = stanza.Pipeline('en')

    for i in pyprind.prog_bar(range(len(multi_str_wikicode))):
        str_wikicode = multi_str_wikicode[i]
        object2entity_ids = dict()
        edge_index,edge_type,index = list(),list(),0
        sentence_num = None
        for j in range(len(str_wikicode)):
            object2entity_ids['sentence-'+ str(j)]  = index
            index +=1
        sentence_num = index
        object2entity_ids['sentence-'+ 'tocken']  = index
        index +=1

        for j in range(len(str_wikicode)):
            ids_,text_ = object2entity_ids['sentence-'+ str(j)],str_wikicode[j]
            doc = nlp(text_)
            # create word index on object2entity_ids
            object2entity_ids['word-' + 'root-' + str(ids_)] = index
            index +=1
            for sent in doc.sentences:
                for word_doc in sent.words:
                    word = word_doc.text
                    if 'word-' + str(word) not in object2entity_ids:
                        object2entity_ids['word-' + str(word)] = index
                        index +=1
                    # construct graph sentence to word (bi-direct)
                    word_ids_ = object2entity_ids['word-' + str(word)]
                    edge_index.append([ids_,word_ids_])
                    edge_type.append(0)
                    edge_index.append([word_ids_,ids_])
                    edge_type.append(1)

            # construct graph word to word (single-direct)
            for sent in doc.sentences:
                for word_doc in sent.words:
                    # Dependency_Parsing
                    word = word_doc.text
                    if word_doc.head > 0:
                        word2 = sent.words[word_doc.head-1].text
                    else:
                        word2 = 'root-' + str(ids_)
                    edge_name = word_doc.deprel

                    word_ids_ = object2entity_ids['word-' + str(word)]
                    word2_ids_ = object2entity_ids['word-' + str(word2)]
                    edge_index.append([word2_ids_,word_ids_])
                    edge_type.append(2)
        object2entity_ids_list.append(object2entity_ids)
        edge_index_list.append(edge_index)
        edge_type_list.append(edge_type)
        index_list.append(index)
        sentence_ids = [k for k in range(sentence_num)]

        sentence_ids_list.append(sentence_ids)
    graph_info = [object2entity_ids_list,edge_index_list,edge_type_list,index_list,sentence_ids_list]
    return graph_info

def Word2Entity2Type(multi_str_wikicode,graph_info):
    object2entity_ids_list = graph_info[0]
    edge_index_list = graph_info[1]
    edge_type_list = graph_info[2]
    index_list = graph_info[3]
    sentence_ids_list = graph_info[4]
    node_no_list = list()
    nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')
    for i in pyprind.prog_bar(range(len(multi_str_wikicode))):
        str_wikicode = multi_str_wikicode[i]
        object2entity_ids = object2entity_ids_list[i]
        edge_index = edge_index_list[i]
        edge_type = edge_type_list[i]
        index = index_list[i]
        for j in range(len(str_wikicode)):
            ids_,text_ = object2entity_ids['sentence-'+ str(j)],str_wikicode[j]
            doc = nlp(text_)
            # create concept and its type on object2entity_ids
            for sent in doc.sentences:
                word_list_ = list() 
                for word_doc in sent.words:
                    word = word_doc.text
                    word_list_.append(word)
                for ent in sent.ents: 
                    entity = ent.text
                    entity_type = ent.type
                    if 'entity-' + str(entity) not in object2entity_ids:
                        object2entity_ids['entity-' + str(entity)] = index
                        index +=1
                    if 'entity-' + str(entity_type) not in object2entity_ids:
                        object2entity_ids['entity-' + str(entity_type)] = index
                        index +=1         
                    # construct graph 'word to concept' and 'concept to concept type' (single-direct)
                    concept_ids_ = object2entity_ids['entity-' + str(entity)]
                    concept_type_ids_ = object2entity_ids['entity-' + str(entity_type)]
                    for k in range(len(word_list_)):
                        if word_list_[k] in entity:
                            word_ids_ = object2entity_ids['word-' + str(word_list_[k])]
                            edge_index.append([word_ids_,concept_ids_])
                            edge_type.append(3)
                    edge_index.append([concept_ids_,concept_type_ids_])
                    edge_type.append(4)
        node_no = len(object2entity_ids)
        node_no_list.append(node_no)
        object2entity_ids_list[i] = object2entity_ids
        edge_index_list[i] = edge_index
        edge_type_list[i] = edge_type
    node_num_give_model = 1 + max(len(object2entity_ids_list[i]) for i in range(len(object2entity_ids_list)))
    new_graph_info = [object2entity_ids_list,edge_index_list,edge_type_list,sentence_ids_list,node_no_list,node_num_give_model]
    return new_graph_info

            
#graph_info = Sent2Word2Word_Construct_Graph(multi_str_wikicode)
# new_graph_info = Word2Entity2Type(multi_str_wikicode,graph_info)



## multi-process version


import pyprind
import stanza
import pickle
class Construct_Graph:
    def __init__(self):
        a= 1

    def Sent2Word2Word_Construct_Graph_MPv(self,i):
        object2entity_ids = dict()
        edge_index,edge_type,index = list(),list(),0
        sentence_num = None
        str_wikicode = self.multi_str_wikicode[i]
        for j in range(len(str_wikicode)):
            object2entity_ids['sentence-'+ str(j)]  = index
            index +=1
        sentence_num = index
        object2entity_ids['sentence-'+ 'tocken']  = index
        index +=1
        for j in range(len(str_wikicode)):
            ids_,text_ = object2entity_ids['sentence-'+ str(j)],str_wikicode[j]
            doc = self.doc_list[i][j]
            # create word index on object2entity_ids
            object2entity_ids['word-' + 'root-' + str(ids_)] = index
            index +=1
            for sent in doc.sentences:
                for word_doc in sent.words:
                    word = word_doc.text
                    if 'word-' + str(word) not in object2entity_ids:
                        object2entity_ids['word-' + str(word)] = index
                        index +=1
                    # construct graph sentence to word (bi-direct)
                    word_ids_ = object2entity_ids['word-' + str(word)]
                    edge_index.append([ids_,word_ids_])
                    edge_type.append(0)
                    edge_index.append([word_ids_,ids_])
                    edge_type.append(1)

            # construct graph word to word (single-direct)
            for sent in doc.sentences:
                for word_doc in sent.words:
                    # Dependency_Parsing
                    word = word_doc.text
                    if word_doc.head > 0:
                        word2 = sent.words[word_doc.head-1].text
                    else:
                        word2 = 'root-' + str(ids_)
                    edge_name = word_doc.deprel

                    word_ids_ = object2entity_ids['word-' + str(word)]
                    word2_ids_ = object2entity_ids['word-' + str(word2)]
                    edge_index.append([word2_ids_,word_ids_])
                    edge_type.append(2)
        sentence_ids = [k for k in range(sentence_num)]
        graph_i_info = [object2entity_ids,edge_index,edge_type,index,sentence_ids]
        return graph_i_info

    def Deploy_doc_MPv_2(self,multi_str_wikicode):
        nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')
        doc_list = list()
        for i in pyprind.prog_bar(range(len(multi_str_wikicode))):
            str_wikicode = multi_str_wikicode[i]
            doc_i = list()
            for j in range(len(str_wikicode)):
                text_ = str_wikicode[j]
                doc = nlp(text_)
                doc_i.append(doc)
            doc_list.append(doc_i)
        return doc_list

    def Deploy_doc_MPv_1(self,multi_str_wikicode):
        nlp = stanza.Pipeline('en')
        doc_list = list()
        for i in pyprind.prog_bar(range(len(multi_str_wikicode))):
            str_wikicode = multi_str_wikicode[i]
            doc_i = list()
            for j in range(len(str_wikicode)):
                text_ = str_wikicode[j]
                doc = nlp(text_)
                doc_i.append(doc)
            doc_list.append(doc_i)
        return doc_list

    def Word2Entity2Type_MPv(self,i,str_wikicode,object2entity_ids,edge_index,edge_type,index,doc_i):
        for j in range(len(str_wikicode)):
            ids_,text_ = object2entity_ids['sentence-'+ str(j)],str_wikicode[j]
            doc = doc_i[j]
            # create concept and its type on object2entity_ids
            for sent in doc.sentences:
                word_list_ = list()
                for word_doc in sent.words:
                    word = word_doc.text
                    word_list_.append(word)
                for ent in sent.ents:
                    entity = ent.text
                    entity_type = ent.type
                    if 'entity-' + str(entity) not in object2entity_ids:
                        object2entity_ids['entity-' + str(entity)] = index
                        index +=1
                    if 'entity-' + str(entity_type) not in object2entity_ids:
                        object2entity_ids['entity-' + str(entity_type)] = index
                        index +=1         
                    # construct graph 'word to concept' and 'concept to concept type' (single-direct)
                    concept_ids_ = object2entity_ids['entity-' + str(entity)]
                    concept_type_ids_ = object2entity_ids['entity-' + str(entity_type)]
                    for k in range(len(word_list_)):
                        if word_list_[k] in entity and str(word_list_[k]) in object2entity_ids:
                            word_ids_ = object2entity_ids['word-' + str(word_list_[k])]
                            edge_index.append([word_ids_,concept_ids_])
                            edge_type.append(3)
                    edge_index.append([concept_ids_,concept_type_ids_])
                    edge_type.append(4)
        node_no = len(object2entity_ids)
        graph_i_info = [object2entity_ids,edge_index,edge_type,node_no]
        return graph_i_info

    def main(self,multi_str_wikicode):
        # with open('stock/example_from_test_graph.pickle', 'rb') as file:
        #     example_from_test_graph = pickle.load(file)
        # str_wikicode = example_from_test_graph['str_wikicode']
        # self.multi_str_wikicode = [str_wikicode,str_wikicode,str_wikicode]
        self.multi_str_wikicode = multi_str_wikicode
        self.doc_list = self.Deploy_doc_MPv_1(multi_str_wikicode)
        MPv_1_input_data = [i for i in range(len(multi_str_wikicode))]
        with Pool() as pool:
            graph_info = pool.map(self.Sent2Word2Word_Construct_Graph_MPv, MPv_1_input_data)
        object2entity_ids_list,edge_index_list,edge_type_list,index_list,sentence_ids_list = list(),list(),list(),list(),list()
        for i in range(len(graph_info)):
            object2entity_ids = graph_info[i][0]
            edge_index = graph_info[i][1]
            edge_type = graph_info[i][2]
            index = graph_info[i][3]
            sentence_ids = graph_info[i][4]
            object2entity_ids_list.append(object2entity_ids)
            edge_index_list.append(edge_index)
            edge_type_list.append(edge_type)
            index_list.append(index)
            sentence_ids_list.append(sentence_ids)
        doc_list = self.Deploy_doc_MPv_2(multi_str_wikicode)
        MPv_2_input_data = [(i,multi_str_wikicode[i],object2entity_ids_list[i],edge_index_list[i],edge_type_list[i],index_list[i],doc_list[i]) for i in range(len(multi_str_wikicode))]
        with Pool() as pool:
            graph_info = pool.starmap(self.Word2Entity2Type_MPv, MPv_2_input_data)
        
        object2entity_ids_list,edge_index_list,edge_type_list,node_no_list = list(),list(),list(),list()
        for i in range(len(graph_info)):
            object2entity_ids = graph_info[i][0]
            edge_index = graph_info[i][1]
            edge_type = graph_info[i][2]
            node_no = graph_info[i][3]
            object2entity_ids_list.append(object2entity_ids)
            edge_index_list.append(edge_index)
            edge_type_list.append(object2entity_ids)
            node_no_list.append(index)
        node_num_give_model = 1 + max(len(object2entity_ids_list[i]) for i in range(len(object2entity_ids_list)))
        new_graph_info = [object2entity_ids_list,edge_index_list,edge_type_list,sentence_ids_list,node_no_list,node_num_give_model]
        return  new_graph_info

# from multiprocessing import Pool
# if __name__=="__main__":
#     construct_graph = Construct_Graph()
#     construct_graph.main()

