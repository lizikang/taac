import pandas as pd
import numpy as np
import os


input_path = 'inputs'
output_path = 'outputs'

# get all prediction results
pred_age_list, pred_gender_list = [], []
preds = os.listdir(input_path)

for p in preds:
	if p[-3:] == 'pkl':
		pred = pd.read_pickle(os.path.join(input_path, p))
	elif p[-3:] == 'npy':
		pred = np.load(os.path.join(input_path, p))
		
	if p[:3] == 'age':
			pred_age_list.append(pred)
	elif p[:3] == 'gen':
			pred_gender_list.append(pred)

	print('add file {}, shape: {}'.format(p, pred.shape))

print('length of pred_age_list: {}'.format(len(pred_age_list)))
print('length of pred_gender_list: {}'.format(len(pred_gender_list)))

# generate the merged prediction
merged_age = np.mean(np.stack(pred_age_list, 1), 1)
np.save(os.path.join(output_path, 'age.npy'), merged_age)
print('the age.npy has been saved!')

merged_gender = np.mean(np.stack(pred_gender_list, 1), 1)
np.save(os.path.join(output_path, 'gender.npy'), merged_gender)
print('the gender.npy has been saved!')

# save the submission result
user = np.arange(3000001, 4000001)
age = np.argmax(merged_age, 1) + 1
gender = np.argmax(merged_gender, 1) + 1

df = pd.DataFrame({'user_id':user, 'predicted_age':age, 'predicted_gender':gender})
df.to_csv(os.path.join(output_path, 'submission.csv'), index=False)
print('the submission.csv has been saved!')
