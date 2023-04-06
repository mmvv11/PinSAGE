# import pickle
# import argparse
#
# parser = argparse.ArgumentParser()
# parser.add_argument('-d', '--dataset-path', type=str)
# args = parser.parse_args()
#
# # print(args.dataset_path)
#
# # Load dataset
# with open(args.dataset_path, 'rb') as f:
#     dataset = pickle.load(f)
#
# data_dict = {
#     'graph': dataset['train-graph'],
#     'val_matrix': None,
#     'test_matrix': None,
#     'item_texts': dataset['item-texts'],
#     'testset': dataset['testset'],
#     'user_ntype': dataset['user-type'],
#     'item_ntype': dataset['item-type'],
#     'user_to_item_etype': dataset['user-to-item-type'],
#     'timestamp': dataset['timestamp-edge-column'],
#     'user_category': dataset['user-category'],
#     'item_category': dataset['item-category']
# }
#
# g =data_dict["graph"]
# # print(g)
# print(g.ndata['id'])
# # print(g.edata)

def clearlog():
    with open("log.txt", 'w') as f:
        f.write("")

def writelog(m):
    m = str(m)
    with open("log.txt", 'a') as f:
        f.write(m + "\n")

clearlog()
writelog(1)
writelog("2")
writelog({"1": "2"})