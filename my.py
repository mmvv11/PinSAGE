import torch
import pickle
import argparse
import torchtext

## TODO 저장된 모델을 불러오기 -> evaluation get_all_emb 등.. 내용 확인

def prepare_dataset(data_dict, args):
    g = data_dict['graph']
    item_texts = data_dict['item_texts']
    user_ntype = data_dict['user_ntype']  # user
    item_ntype = data_dict['item_ntype']  # wine

    # Assign user and movie IDs and use them as features (to learn an individual trainable
    # embedding for each entity)
    # TODO 각 노드의 갯수만큼 id를 부여해준다. 나중에 임베딩 각각 부여해줄 때 필요한듯?
    g.nodes[user_ntype].data['id'] = torch.arange(g.number_of_nodes(user_ntype))
    g.nodes[item_ntype].data['id'] = torch.arange(g.number_of_nodes(item_ntype))
    data_dict['graph'] = g

    # Prepare torchtext dataset and vocabulary
    # TODO 텍스트 처리하는 것 같은데 일단
    if not len(item_texts):
        data_dict['textset'] = None
    else:
        fields = {}
        examples = []
        for key, texts in item_texts.items():
            fields[key] = torchtext.data.Field(include_lengths=True, lower=True, batch_first=True)
        for i in range(g.number_of_nodes(item_ntype)):
            example = torchtext.data.Example.fromlist(
                [item_texts[key][i] for key in item_texts.keys()],
                [(key, fields[key]) for key in item_texts.keys()])
            examples.append(example)

        textset = torchtext.data.Dataset(examples, fields)
        for key, field in fields.items():
            field.build_vocab(getattr(textset, key))
            # field.build_vocab(getattr(textset, key), vectors='fasttext.simple.300d')
        data_dict['textset'] = textset

    return data_dict

# Load dataset
with open("data.pkl", 'rb') as f:
    dataset = pickle.load(f)

data_dict = {
    'graph': dataset['train-graph'],
    'val_matrix': None,
    'test_matrix': None,
    'item_texts': dataset['item-texts'],
    'testset': dataset['testset'],
    'user_ntype': dataset['user-type'],
    'item_ntype': dataset['item-type'],
    'user_to_item_etype': dataset['user-to-item-type'],
    'timestamp': dataset['timestamp-edge-column'],
    'user_category': dataset['user-category'],
    'item_category': dataset['item-category']
}

# Dataset
data_dict = prepare_dataset(data_dict, args)