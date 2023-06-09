import os
import re
import json
import pickle
import argparse
import scipy.sparse as ssp
from collections import defaultdict

from tqdm import tqdm
import pandas as pd
import numpy as np
import dgl
import torch
import torchtext
from builder import PandasGraphBuilder
from data_utils import *

print('Processing Started!')
path = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument('--directory', default=os.path.join(path, 'data'), type=str)
parser.add_argument('--output_path', default="data.pkl", type=str)
args = parser.parse_args()
directory = args.directory
output_path = args.output_path

# User Data
with open(os.path.join(directory, 'user.json')) as f:
    user_json = json.load(f)
users = pd.DataFrame(user_json["data"])

columns = ['userID', 'user_follower_count', 'user_rating_count']
users = users[columns]
users = users.dropna(subset=['userID'])
users['user_feats'] = list(users[['user_follower_count', 'user_rating_count']].values)

# Wine Data
with open(os.path.join(directory, 'wine.json')) as f:
    item_json = json.load(f)
items = pd.DataFrame(item_json["data"])
items = items.dropna()

columns = ['wine_id', 'name', 'rating_average', 'body', 'acidity_x', 'alcohol', 'grapes_id']
items = items[columns]
items = items.dropna(subset=['wine_id', 'grapes_id'])

items['grapes_id'] = [i[1] for i in items['grapes_id']]
items['wine_feats'] = list(items[['rating_average', 'body', 'acidity_x', 'alcohol']].values)

# Rating Data
# 8:2
columns = ['userID', 'wine_id', 'rating_per_user']

with open(os.path.join(directory, 'train.json')) as f:
    train = json.load(f)
train = pd.DataFrame(train['data'])

'''
* Like
I thought about whether it would be a good choice to recommend an item that the user did not evaluate well.
If you want to learn regardless of the rating, you can remove the code related to like.

We set the standard for like as a rating of 3.
I think it would be good to set this part according to your own thoughts.
'''
train['like'] = [1 if x >= 3 else 0 for x in train['rating_per_user']]
train = train[train['like'] == 1]
train = train[columns]

with open(os.path.join(directory, 'test.json')) as f:
    test = json.load(f)
test = pd.DataFrame(test['data'])

test['like'] = [1 if x >= 3 else 0 for x in test['rating_per_user']]
test = test[test['like'] == 1]
test = test[columns]

ratings = pd.concat([train, test], axis=0, ignore_index=True)
## TODO 유저 필터로 데이터 갯수 조절 가능할 듯. 초기값은 v > 1.. 인데 나는 ==7으로 해보겠음.
user_filter = [k for k, v in ratings['userID'].value_counts().items() if v == 7]
ratings = ratings[ratings['userID'].isin(user_filter)]

# Build Graph
# 아이템, 유저 DB에 존재하는 rating만 사용
user_intersect = set(ratings['userID'].values) & set(users['userID'].values)
item_intersect = set(ratings['wine_id'].values) & set(items['wine_id'].values)

new_users = users[users['userID'].isin(user_intersect)]
new_items = items[items['wine_id'].isin(item_intersect)]
new_ratings = ratings[ratings['userID'].isin(user_intersect) & ratings['wine_id'].isin(item_intersect)]
new_ratings = new_ratings.sort_values('userID')

label = []
for userID, df in new_ratings.groupby('userID'):
    idx = int(df.shape[0] * 0.8)
    timestamp = [0] * df.shape[0]
    timestamp = [x if i < idx else 1 for i, x in enumerate(timestamp)]
    label.extend(timestamp)
new_ratings['timestamp'] = label

# Build graph
graph_builder = PandasGraphBuilder()
graph_builder.add_entities(new_users, 'userID', 'user')
graph_builder.add_entities(new_items, 'wine_id', 'wine')
graph_builder.add_binary_relations(new_ratings, 'userID', 'wine_id', 'rated')
graph_builder.add_binary_relations(new_ratings, 'wine_id', 'userID', 'rated-by')
g = graph_builder.build()

# Assign features.
node_dict = {
    'user': [new_users, ['userID', 'user_feats'], ['cat', 'int']],
    'wine': [new_items, ['wine_id', 'grapes_id', 'wine_feats'], ['cat', 'cat', 'int']]
}
edge_dict = {
    'rated': [new_ratings, ['rating_per_user', 'timestamp']],
    'rated-by': [new_ratings, ['rating_per_user', 'timestamp']]
}

for key, (df, features, dtypes) in node_dict.items():
    for value, dtype in zip(features, dtypes):
        # key = 'user' or 'wine'
        # value = 'user_follower_count' 등등
        if dtype == 'int':
            array = np.array([i for i in df[value].values])
            g.nodes[key].data[value] = torch.FloatTensor(array)
        elif dtype == 'cat':
            g.nodes[key].data[value] = torch.LongTensor(df[value].astype('category').cat.codes.values)

for key, (df, features) in edge_dict.items():
    for value in features:
        g.edges[key].data[value] = torch.LongTensor(df[value].values.astype(np.float32))

# 실제 ID와 카테고리 ID 딕셔너리
user_cat = new_users['userID'].astype('category').cat.codes.values
item_cat = new_items['wine_id'].astype('category').cat.codes.values

user_cat_dict = {k: v for k, v in zip(user_cat, new_users['userID'].values)}
item_cat_dict = {k: v for k, v in zip(item_cat, new_items['wine_id'].values)}

# Label
val_dict = defaultdict(set)
for userID, df in new_ratings.groupby('userID'):
    val_dict[userID] = set(df[df['timestamp'] == 1]['wine_id'].values)

# Build title set
textual_feature = {'name': items['name'].values}

# Dump the graph and the datasets
dataset = {
    'train-graph': g,
    'user-data': new_users,
    'item-data': new_items,
    'rating-data': new_ratings,
    'val-matrix': None,
    'test-matrix': torch.LongTensor([[0]]),
    'testset': val_dict,
    'item-texts': textual_feature,
    'item-images': None,
    'user-type': 'user',
    'item-type': 'wine',
    'user-category': user_cat_dict,
    'item-category': item_cat_dict,
    'user-to-item-type': 'rated',
    'item-to-user-type': 'rated-by',
    'timestamp-edge-column': 'timestamp'}

with open(output_path, 'wb') as f:
    pickle.dump(dataset, f)

print('Processing Completed!')
