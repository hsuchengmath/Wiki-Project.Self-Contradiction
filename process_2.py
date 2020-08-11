## segment sentence by stanza
#import stanza
import mwparserfromhell
import re
import pickle
import pyprind
from nltk.tokenize import sent_tokenize

# reload a file to a variable
with open('stock/self_contradictory_XY_1_v2.pickle', 'rb') as file:
    self_contradictory_XY_1 =pickle.load(file)

X = self_contradictory_XY_1['X']
Y = self_contradictory_XY_1['Y']
title = self_contradictory_XY_1['title']

#nlp = stanza.Pipeline('en')
X_standard = list()
for i in pyprind.prog_bar(range(len(X))):
    text = X[i]
    wikicode = mwparserfromhell.parse(text)
    templates = wikicode.filter_templates()

    str_wikicode = str(wikicode)
    for j in range(len(templates)):
        str_wikicode = str_wikicode.replace(str(templates[j]),' ')
    str_wikicode = str_wikicode.replace('[[','')
    str_wikicode = str_wikicode.replace(']]','')
    str_wikicode = sent_tokenize(str_wikicode)
    # doc = nlp(str_wikicode)
    # artcile = list()
    # for j, sent in enumerate(doc.sentences):
    #     sentence = [sent.tokens[k].text for k in range(len(sent.tokens))]
    #     sentence = ' '.join(sentence)
    #     artcile.append(sentence)
    artcile = list()
    for j in range(len(str_wikicode)):
        sent = ' '.join(re.sub('[^a-zA-Z]',' ',str_wikicode[j]).split())
        if len(sent.split()) != 0:
            artcile.append(sent)
    X_standard.append(artcile)




a_dict = {'X':X_standard ,'Y':Y,'title':title}




##pickle a variable to a file
file = open('stock/self_contradictory_XY_2_v2.pickle', 'wb')
pickle.dump(a_dict, file)
file.close()
