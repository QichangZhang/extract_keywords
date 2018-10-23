# -*- encoding:utf-8 -*-
'''
1.分词-去停用词和非中文字符
2.采用textrank提取文本的关键词
3.采用tdidf提取文章排名前5的词，并和语料中的热词进行比较（将整个语料进行tfidf计算热词），
如果词和语料中的热词匹配，则作为一个热词
4.将2和3的结果合并作为文档的关键词
'''

import os
import re
import pickle
import itertools
import codecs
import jieba
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from textrank4zh  import TextRank4Sentence, TextRank4Keyword
from collections import Counter
key_word = TextRank4Keyword()
stop_words = set(open(os.path.join('CNstopwords.txt'), 'r', encoding= 'utf-8_sig').read().split())

def ext_keywords_jieba(text):
    keywords = analyse.extract_tags(text, topK= 10, withWeight= False, allowPOS= ())
    print('keyword', [keyword for keyword in keywords])
    return keywords

def clean_data(s):
    words = jieba.cut(s)
    filter_words = []
    for w in words:
        if w in stop_words:
            filter_words.append(' ')  # 去停用词
        else:
            filter_words.append(w)
    s = ' '.join(filter_words)
    s = re.sub("[-+]?\d+[\.]?\d*", ' ', s)  # 替换数字
    s = re.sub(u"[^\u4e00-\u9fa5]+", ' ', s)  # 替换非中文字符
    return s

def extract_keywords(text):
    keywords = []
    word = TextRank4Keyword()
    word.analyze(text, window=4, lower=True)
    w_list = word.get_keywords(num=15, word_min_len=2)
    for w in w_list:
        keywords.append(w.word)
    return keywords

#基于tfidf的热词不正常
def hot_word_accumulate(text):
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform([text]))
    weights = tfidf.toarray()[0]
    words = vectorizer.get_feature_names()
    keywords = zip(words, weights)
    keywords = sorted(keywords, key = lambda keyword: keyword[1])[:5]
    word = []
    for keyword in keywords:
        word.append(keyword[0])
    return word

class PMI:
    def __init__(self, document):
        self.document = document
        self.pmi = {}
        self.miniprobability = float(1.0) / document.__len__()
        self.minitogether = float(0)/ document.__len__()
        self.set_word = self.getset_word()

    def calcularprobability(self, document, wordlist):

        """
        :param document:
        :param wordlist:
        :function : 计算单词的document frequency
        :return: document frequency
        """

        total = document.__len__()
        number = 0
        for doc in self.set_word:
            if set(wordlist[0]).issubset(doc):
                number += 1
        percent = float(number)/total
        return percent

    def togetherprobablity(self, document, wordlist1, wordlist2):

        """
        :param document:
        :param wordlist1:
        :param wordlist2:
        :function: 计算单词的共现概率
        :return:共现概率
        """

        joinwordlist = wordlist1 + wordlist2
        percent = self.calcularprobability(document, joinwordlist)
        return percent

    def getset_word(self):

        """
        :function: 得到document中的词语词典
        :return: 词语词典
        """
        list_word = self.document.split(' ')
        set_word = []
        for w in list_word:
            if set_word.count(w) == 0:
                set_word.append(w)
        return set_word

    def get_dict_frq_word(self):

        """
        :function: 对词典进行剪枝,剪去出现频率较少的单词
        :return: 剪枝后的词典
        """
        dict_frq_word = {}
        for i in range(0, self.set_word.__len__(), 1):
            list_word=[]
            list_word.append(self.set_word[i])
            probability = self.calcularprobability(self.document, list_word)
            if probability > self.miniprobability:
                dict_frq_word[self.set_word[i]] = probability
        return dict_frq_word

    def calculate_nmi(self, joinpercent, wordpercent1, wordpercent2):
        """
        function: 计算词语共现的nmi值
        :param joinpercent:
        :param wordpercent1:
        :param wordpercent2:
        :return:nmi
        """
        return (joinpercent)/(wordpercent1*wordpercent2)

    def get_pmi(self):
        """
        function:返回符合阈值的pmi列表
        :return:pmi列表
        """
        dict_pmi = {}
        dict_frq_word = self.get_dict_frq_word()
        for word1 in dict_frq_word:
            wordpercent1 = dict_frq_word[word1]
            for word2 in dict_frq_word:
                if word1 == word2:
                    continue
                wordpercent2 = dict_frq_word[word2]
                list_together=[]
                list_together.append(word1)
                list_together.append(word2)
                together_probability = self.calcularprobability(self.document, list_together)
                if together_probability > self.minitogether:
                    string = word1 + ',' + word2
                    dict_pmi[string] = self.calculate_nmi(together_probability, wordpercent1, wordpercent2)
        return dict_pmi

def get_pmi_word(dict_pmi, text, keywords):
    keyword_dict = {}

    for keyword in keywords:
        score = 0
        for word in text.split(' '):
            key1 = word + ',' + keyword
            key2 = keyword + ',' + word
            if key1 in dict_pmi.keys():
                if score < dict_pmi[key1]:
                    score = dict_pmi[key1]
                    keyword_dict[keyword] = word
            elif key2 in dict_pmi.keys():
                if score < dict_pmi[key2]:
                    score = dict_pmi[key1]
                    keyword_dict[keyword] = word
    return keyword_dict