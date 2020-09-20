import os
import numpy as np
import pandas as pd
from tqdm import tqdm


input_path = '.'
output_path = '.'
max_seq_len = 150
min_freq = 3


# transform data
def transform(data):
	data = data.replace('\\N', 0)
	data = data.astype('int64')
	data = data.sort_values(['user_id', 'time'])
	data = data.groupby('user_id').tail(max_seq_len)
	return data


# read train preliminary files
print('begin to read train preliminary files...')
train_path = os.path.join(input_path, 'train_preliminary')
train_user = pd.read_csv(os.path.join(train_path, 'user.csv'))
train_click_log = pd.read_csv(os.path.join(train_path, 'click_log.csv'))
train_ad = pd.read_csv(os.path.join(train_path, 'ad.csv'))

# read train sime-final files
print('begin to read train sime-final files...')
train1_path = os.path.join(input_path, 'train_semi_final')
train1_user = pd.read_csv(os.path.join(train1_path, 'user.csv'))
train1_click_log = pd.read_csv(os.path.join(train1_path, 'click_log.csv'))
train1_ad = pd.read_csv(os.path.join(train1_path, 'ad.csv'))

# concat train_preliminary and train_semi_final file
print('begin to concat train_preliminary and train_semi_final file...')
train_user = pd.concat([train_user, train1_user])
train_click_log = pd.concat([train_click_log, train1_click_log])
train_ad = pd.concat([train_ad, train1_ad])

# merge age and gender, merge all train files, transform data
print('begin to merge and transform train data...')
train_user['agender'] = train_user['age'] + (train_user['gender']-1)*10 - 1
train_all_data = pd.merge(train_click_log, train_ad, on=['creative_id'], how='left')
train_all_data = pd.merge(train_all_data, train_user, on=['user_id'], how='left')
train_all_data = transform(train_all_data)


# read test files
print('begin to read test files...')
test_path = os.path.join(input_path, 'test')
test_click_log = pd.read_csv(os.path.join(test_path, 'click_log.csv'))
test_ad = pd.read_csv(os.path.join(test_path, 'ad.csv'))

# merge all test files, transform data
print('begin to merge and transform test data...')
test_all_data = pd.merge(test_click_log, test_ad, on=['creative_id'], how='left')
test_all_data = transform(test_all_data)


# drop feature ids with low frequency
print('begin to drop low frequency feature ids...')
features_to_drop = ['creative_id', 'ad_id', 'product_id', 'product_category', 'advertiser_id', 'industry']
for fea in features_to_drop:
	if fea not in ['creative_id', 'ad_id']:
		min_freq = 2

	id_counts = pd.concat([train_all_data[fea], test_all_data[fea]]).nunique()
	print('before dropping low frequency {}, total id numbers: {}'.format(fea, id_counts))

	freq = pd.concat([train_all_data[fea], test_all_data[fea]]).value_counts()
	low_freq_ids = freq[freq<min_freq].index
	train_all_data.loc[train_all_data[fea].isin(low_freq_ids), fea] = 0
	test_all_data.loc[test_all_data[fea].isin(low_freq_ids), fea] = 0

	id_counts = pd.concat([train_all_data[fea], test_all_data[fea]]).nunique()
	print('after dropping low frequency {}, total id numbers: {}'.format(fea, id_counts))


# map original feature id to new id (1,2,3,..,n)
print('begin to map feature id...')
features_to_map = ['creative_id', 'click_times', 'ad_id', 'product_id', 'product_category', 'advertiser_id', 'industry']
for fea in features_to_map:
	original_ids = pd.concat([train_all_data[fea], test_all_data[fea]]).unique().tolist()
	new_ids = list(range(1, len(original_ids)+1))
	map_dict = dict(zip(original_ids, new_ids))

	train_all_data[fea] = train_all_data[fea].map(map_dict)
	test_all_data[fea] = test_all_data[fea].map(map_dict)


# save the train and test file
print('begin to save train and test data...')
train_all_data.to_csv(os.path.join(output_path, 'train_all_data.csv'), index=False)
test_all_data.to_csv(os.path.join(output_path, 'test_all_data.csv'), index=False)
print('Done!')
