# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 18:59:31 2021

@author: suryaravi

reference URL is https://www.pluralsight.com/guides/natural-language-processing-named-entity-recognition
"""

import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
#nltk.download('wordnet')  #download if using this module for the first time
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
#nltk.download('stopwords')    #download if using this module for the first time
from nltk.tokenize import word_tokenize
from nltk import word_tokenize, pos_tag, ne_chunk
#nltk.download('words')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('punkt')
#nltk.download('maxent_ne_chunker')

#textexample = "Avengers: Endgame is a 2019 American superhero film based on the Marvel Comics superhero team the Avengers, produced by Marvel Studios and distributed by Walt Disney Studios Motion Pictures. The movie features an ensemble cast including Robert Downey Jr., Chris Evans, Mark Ruffalo, Chris Hemsworth, and others. (Source: wikipedia)."
textexample = "Prime Minister Jacinda Ardern has claimed that New Zealand had won a big battle over the spread of coronavirus. Her words came as the country begins to exit from its lockdown"
print(textexample)

sentences = nltk.sent_tokenize(textexample)
tokenized_sentence = [nltk.word_tokenize(sent) for sent in sentences]
tokenized_sentence 

pos_tagging_sentences = [nltk.pos_tag(sent) for sent in tokenized_sentence]

def preprocess(text):
    text = nltk.word_tokenize(text)
    text = nltk.pos_tag(text)
    return text

processed_text = preprocess(textexample)
processed_text

res_chunk = ne_chunk(processed_text)

for x in str(res_chunk).split('\n'):
    if '/NN' in x:
        print(x)




