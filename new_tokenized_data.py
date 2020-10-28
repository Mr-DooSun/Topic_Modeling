# -*- coding: utf-8 -*-
import re
from tika import parser
from konlpy.tag import Mecab
import csv

mecab = Mecab()

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
		sentence = rawText['content'].strip()
		pattern = '[^\w\s]'
		sentence = re.sub(pattern=pattern, repl='', string=sentence)

		# print(sentence)

		for _ in range(10) :
			sentence.replace('  ',' ')

		sentence = sentence.split('\n\n')
		text = get_nouns(sentence)
		processed_data.append(text.split()) 

	return processed_data

def save_processed_data(processed_data) :
	with open('new_tokenized_data.csv','w',newline='',encoding='utf-8') as f:
		writer = csv.writer(f)
		for data in processed_data:
			writer.writerow(data)

if __name__ == '__main__' :
	data = ['data/삼성전자 반기보고서.pdf','data/CJ 반기보고서.pdf','data/LG 반기보고서.pdf','data/현대자동차 반기보고서.pdf','data/한화 반기보고서.pdf']

	# data = ['data/CJ 반기보고서.pdf']
	processed_data = data_parsing(data)

	# print(processed_data)

	save_processed_data(processed_data)