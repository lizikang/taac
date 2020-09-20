import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


input_path = '../data'
output_path = '.'


print('begin to read train and test data')
train_all_data = pd.read_csv(os.path.join(input_path, 'train_all_data.csv'))
test_all_data = pd.read_csv(os.path.join(input_path, 'test_all_data.csv'))


features = ['creative_id']
for fea in features:
	print('begin to process feature {}'.format(fea))

	print('\tbegin to generate corpus')
	all_data = pd.concat([train_all_data[['user_id', fea]], test_all_data[['user_id', fea]]])
	all_data[fea] = all_data[fea].astype('str')
	corpus = list(all_data.groupby('user_id').apply(lambda x: ' '.join(x[fea])))

	print('\tbegin to generate tfidf scores')
	vectorizer = TfidfVectorizer()
	tfidf_matrix = vectorizer.fit_transform(corpus)
	word2id_dict = vectorizer.vocabulary_
	print('shape of tfidf_matrix: {}'.format(tfidf_matrix.shape))
	print(len(word2id_dict))
	print(pd.concat([train_all_data[fea], test_all_data[fea]]).nunique())
	print(word2id_dict.get('1', None))
	print(word2id_dict.get(1, None))

	print('\tbegin to map tfidf scores')
	tfidf_list = []
	for i, doc in enumerate(corpus):
		word_list = doc.split()
		id_list = [word2id_dict[word] for word in word_list]
		tdidf = list(tfidf_matrix[i].toarray()[0][id_list])
		tfidf_list.extend(tfidf)

	print('\t{}'.format(len(tfidf_list)-len(train_all_data)-len(test_all_data)))
	train_all_data[fea+'_tfidf'] = tfidf_list[:len(train_all_data)]
	test_all_data[fea+'_tfidf'] = tfidf_list[len(train_all_data):]

print('begin to save train and test data')
train_all_data.to_csv(os.path.join(output_path, 'train_all_data.csv'), index=False)
test_all_data.to_csv(os.path.join(output_path, 'test_all_data.csv'), index=False)
