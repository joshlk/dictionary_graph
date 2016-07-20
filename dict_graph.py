#!/usr/bin/env python2

import spacy
import igraph
import ujson
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

import networkx as nx

__author__ = 'Josh Levy-Kramer'

with open('dictionary/dictionary.json') as f:
    en_dict_json = ujson.load(f)

en_dict = en_dict_json.items()      # As tuple
en_dict = pd.DataFrame(en_dict, columns=['word', 'definition'])

en_dict['word'] = en_dict['word'].str.lower()

nlp_en = spacy.load('en')
en_dict['definitions_parsed'] = list(nlp_en.pipe(en_dict['definition'], batch_size=100000, n_threads=10))      # Takes around 30s
en_dict['word_parsed'] = list(nlp_en.pipe(en_dict['word'], batch_size=100000, n_threads=10))

# Convert to edge list and filter out punct and lemmas
words_all = []
lemmas_all = []
for ix, row in en_dict.iterrows():
    word_doc = row['word_parsed']
    if (len(word_doc) > 1) or word_doc[0].is_stop:
        continue

    definition_doc = row['definitions_parsed']
    #lemmas = [t.lemma_ for t in definition_doc if not t.is_stop and not t.is_punct]
    lemmas = []
    for term in definition_doc:
        if not term.is_stop and not term.is_punct:
            try:
                lemmas.append(str(term))    # If it cant be non-unicode ignor
            except:
                pass

    lemmas_all.extend(lemmas)
    words_all.extend([str(word_doc[0])] * len(lemmas))


# Convert to graph
g = igraph.Graph.TupleList(zip(words_all, lemmas_all), directed=True)


# Obtain largest connected component
components = g.components()
components_subgraphs = components.subgraphs()
g_largest = np.argmax(components.sizes())
g_largest = components_subgraphs[g_largest]

# betweeness
betweenness = g_largest.betweenness()
betweenness = pd.DataFrame({'word': g_largest.vs.get_attribute_values('name'), 'betweenness': betweenness})
betweenness = betweenness.sort_values('betweenness', ascending=False)

# pagerank
pagerank = g_largest.pagerank()
pagerank = pd.DataFrame({'word': g_largest.vs.get_attribute_values('name'), 'pagerank': pagerank})
pagerank = pagerank.sort_values('pagerank', ascending=False)



#
#
#
#
#
#
#
#

#
# g = igraph.Graph()
#
#
# # Make sure theres a connected component
# components = g.components()
#
# # Betweenness
# betweenness = g.betweenness()
# betweenness = pd.DataFrame({'word': g.vs.get_attribute_values('name'), 'betweenness': betweenness})
# betweenness = betweenness.sort_values('betweenness', ascending=False)
#
# #
# #
# # Obtain lemma array
# en_dict['definitions_lemma'] = en_dict['definitions_parsed'].apply(lambda doc: doc.to_array([spacy.attrs.LEMMA]))
# en_dict['words_lemma'] = en_dict['word_parsed'].apply(lambda doc: doc.to_array([spacy.attrs.LEMMA]))
#
# # Remove 'words' that are made up of more than one word
# en_dict['words_lemma_len'] = en_dict['words_lemma'].apply(lambda a: len(a))
# en_dict_1 = en_dict[en_dict['words_lemma_len'] == 1]
#
# # Convert to edge list
# edge_list = []
# for ix, row in en_dict_1.iterrows():
#     definition = row['definitions_lemma']
#     word = row['words_lemma']
#     df = pd.DataFrame()
#     df['definition'] = definition.flatten()
#     df['word'] = word[0][0]
#     edge_list.append(df)
# edge_list = pd.concat(edge_list)
#
# # Convert to graph
# g = igraph.Graph.TupleList(edge_list.values.tolist(), directed=True)
#
# # Betweenness
# betweenness = g.betweenness()
# betweenness = pd.DataFrame({'word_id': g.vs., 'betweenness': betweenness})
