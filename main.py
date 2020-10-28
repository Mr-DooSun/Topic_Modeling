# -*- coding: utf-8 -*-
from gensim import corpora
from gensim.models import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.models.callbacks import CoherenceMetric
from gensim.models.callbacks import PerplexityMetric

from tqdm import tqdm

import matplotlib.pyplot as plt

import logging
import re

from tika import parser
from konlpy.tag import Mecab

import csv

mecab = Mecab()

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
	coherence_values = []
	model_list = []
	
	for num_topics in range(start, limit, step):
		model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
		model_list.append(model) 
		coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='u_mass')
		coherence_values.append(coherencemodel.get_coherence())

	return model_list, coherence_values

def find_optimal_number_of_topics(dictionary, corpus, processed_data):
	limit = 40;
	start = 2;
	step = 3;

	model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=processed_data, start=start, limit=limit, step=step)

	x = range(start, limit, step)
	plt.plot(x, coherence_values)
	plt.xlabel('Num Topics')
	plt.ylabel('Coherence score')
	plt.legend(('coherence_values'), loc ='best')
	plt.show()

def get_nouns(sentence) :
	text = ""

	for word in sentence :
		tagged = mecab.pos(word)
		nouns = [s for s, t in tagged if t in ['SL','NNG','NNP'] and len(s) > 1]
		if len(nouns) > 0 :
			for t in nouns :
				text += t+" "

	return text

def data_parsing(data):
	processed_data = []
	for file in data :
		rawText = parser.from_file(file)
		sentence = rawText['content'].replace('.','').strip()
		pattern = '[^\w\s]'
		sentence = re.sub(pattern=pattern, repl='', string=sentence)

		for _ in range(10) :
			sentence.replace('  ',' ')

		sentence = sentence.strip().split('\n\n')
		text = get_nouns(sentence)
		processed_data.append(text.split()) 

	return processed_data

if __name__ == '__main__' :
	data = ['data/삼성전자 반기보고서.pdf','data/CJ 반기보고서.pdf','data/LG 반기보고서.pdf','data/현대자동차 반기보고서.pdf','data/한화 반기보고서.pdf']

	# processed_data = data_parsing(data)
	processed_data = [sent.strip().split(",") for sent in tqdm(open('new_tokenized_data.csv', 'r', encoding='utf-8').readlines())]

	dictionary = corpora.Dictionary(processed_data)
	
	print(dictionary)

	# dictionary.filter_extremes(no_below=1, no_above=0.5)
	corpus = [dictionary.doc2bow(text) for text in processed_data]
	print('Number of unique tokens: %d' % len(dictionary))
	print('Number of documents: %d' % len(corpus))

	logging.basicConfig(format='%(asctime)s : %=[(levelname)s : %(message)s', level=logging.INFO)

	# # 최적의 토픽 수 찾기
	find_optimal_number_of_topics(dictionary, corpus, processed_data)

