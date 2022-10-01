

# Load csv file to dict
import csv

csv_path = '/Users/bytedance/Downloads/Batch_4820165_batch_results.csv' # 200 per worker
csv_path2 = '/Users/bytedance/Downloads/Batch_4820176_batch_results.csv' # 70 per worker

data_dict = {}
with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['WorkerId'] not in data_dict:
            data_dict[row['WorkerId']] = 0
        data_dict[row['WorkerId']] += 1

a=1

data_dict2 = {}
with open(csv_path2, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['WorkerId'] not in data_dict2:
            data_dict2[row['WorkerId']] = 0
        data_dict2[row['WorkerId']] += 1

a=1