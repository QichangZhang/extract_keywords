from jieba.analyse import *
from textrank4zh import TextRank4Keyword
from pyltp import Segmentor, Postagger, NamedEntityRecognizer, Parser
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import data_tools
import codecs
import nltk
import jieba
import pandas as pd
import os
from flask import Flask,jsonify,request
app = Flask(__name__)
@app.route('/api/keywords', methods = ['post'])
def process():
    try:
        text = request.files['text_path'].read()
    except Exception:
        return '未输入文件'
    text = data_tools.clean_data(text)
    ext_words = data_tools.extract_keywords(text)
    dict_pmi = data_tools.PMI(text).get_pmi()
    pmi_words = data_tools.get_pmi_word(dict_pmi, text, ext_words)
    keywords = []
    for w in ext_words:
        keywords.append(w)
    for w in ext_words:
        if w in pmi_words.keys():
            keywords.append(pmi_words[w])
    keywords = set(keywords)
    return str(keywords)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8131, debug=True)