import pandas as pd
## csv part.
import pyprind
import mwparserfromhell
def Format_csv2XY(path):
    X,Y,title = list(),list(),list()
    df = pd.read_csv(path)
    page_title = list(df['page_title'])
    revision_text = list(df['revision_text'])  
    for i in pyprind.prog_bar(range(len(revision_text))):
        text = revision_text[i]
        title_i = page_title[i]
        if isinstance(text,str) is True and len(text.split()) !=0:
            wikicode = mwparserfromhell.parse(text)
            templates = wikicode.filter_templates()
            is_pos = False
            for j in range(len(templates)):
                if 'Self-contradictory' in templates[j]:
                    is_pos = True
            if is_pos:
                X.append(str(text))
                title.append(title_i)
                Y.append(1)
            else:
                X.append(str(text))
                title.append(title_i)
                Y.append(0)         
    return  X,Y,title

## jsonl part
import pyprind
import json


def Format_json2XY(path):
    with open(path, 'r') as jsonlfile:
        jsonlfile = list(jsonlfile)
    X,Y,title = list(),list(),list()
    for i in pyprind.prog_bar(range(len(jsonlfile))):
        a = json.loads(jsonlfile[i])
        text = a['povVersion']
        solve_text = a['solvedpovVersion']
        pageTitle = a['pageTitle']
        wikicode = mwparserfromhell.parse(text)
        templates = wikicode.filter_templates()
        is_pos_1 = False
        for j in range(len(templates)):
            if 'Self-contradictory' in templates[j]: 
                is_pos_1 = True
        wikicode = mwparserfromhell.parse(solve_text)
        templates = wikicode.filter_templates()
        is_pos_2 = False
        for j in range(len(templates)):
            if 'Self-contradictory' in templates[j]: 
                is_pos_2 = True
        if is_pos_1 is True and len(text.split()) != 0:
            X.append(str(text))
            title.append(pageTitle)
            Y.append(1)
        else:
            X.append(str(text))
            title.append(pageTitle)
            Y.append(0)            
        if is_pos_2 is True and len(solve_text.split()) != 0:
            X.append(str(solve_text))
            title.append(pageTitle)
            Y.append(1)
        else:
            X.append(str(solve_text))
            title.append(pageTitle)
            Y.append(0)            
    return X,Y,title 




json_path = 'dataset/' + 'solved.jsonl'
selfC_path = 'dataset/selfC.csv'
solvedSelfC_path = 'dataset/solvedSelfC.csv'

json_X,json_Y,json_title  = Format_json2XY(json_path)

selfC_X,selfC_Y,selfC_title = Format_csv2XY(selfC_path)
solvedSelfC_X,solvedSelfC_Y,solvedSelfC_title = Format_csv2XY(solvedSelfC_path)


X = json_X + selfC_X + solvedSelfC_X
Y = json_Y + selfC_Y + solvedSelfC_Y
title = json_title + selfC_title + solvedSelfC_title



def Random_Filter_Pos_Neg(X,Y,title):
    pos_ids,neg_ids = list(),list()
    for i in range(len(Y)):
        if Y[i] == 1:
            pos_ids.append(i)
        elif Y[i] == 0:
            neg_ids.append(i)
        else:
            print('[ERROR!!]')
    import random
    data_num = 1000 #total is 2000 (pos + neg)
    selected_pos_ids = random.sample(pos_ids,data_num)
    selected_neg_ids = random.sample(neg_ids,data_num)
    selected_X,selected_Y,selected_title = list(),list(),list()
    for i in range(len(selected_pos_ids)):
        selected_X.append(X[selected_pos_ids[i]])
        selected_Y.append(Y[selected_pos_ids[i]])
        selected_title.append(title[selected_pos_ids[i]])
    for i in range(len(selected_neg_ids)):
        selected_X.append(X[selected_neg_ids[i]])
        selected_Y.append(Y[selected_neg_ids[i]])
        selected_title.append(title[selected_neg_ids[i]])
    return selected_X,selected_Y,selected_title


selected_X,selected_Y,selected_title = Random_Filter_Pos_Neg(X,Y,title)





import pickle

a_dict = {'X':selected_X ,'Y':selected_Y,'title':selected_title}

# pickle a variable to a file


file = open('stock/self_contradictory_XY_1_v2.pickle', 'wb')
pickle.dump(a_dict, file)
file.close()
