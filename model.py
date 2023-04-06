import pickle
import argparse

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torchtext
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import layers
import sampler as sampler_module
import evaluation

def clearlog():
    with open("log.txt", 'w') as f:
        f.write("")

def writelog(m):
    m = str(m)
    with open("log.txt", 'a') as f:
        f.write(m + "\n")

class PinSAGEModel(nn.Module):
    def __init__(self, full_graph, ntype, textsets, hidden_dims, n_layers):
        super().__init__()

        self.proj = layers.LinearProjector(full_graph, ntype, textsets, hidden_dims)
        self.sage = layers.SAGENet(hidden_dims, n_layers)
        self.scorer = layers.ItemToItemScorer(full_graph, ntype)

    def forward(self, pos_graph, neg_graph, blocks):
        h_item = self.get_repr(blocks)
        pos_score = self.scorer(pos_graph, h_item)
        neg_score = self.scorer(neg_graph, h_item)
        return (neg_score - pos_score + 1).clamp(min=0)

    def get_repr(self, blocks):
        h_item = self.proj(blocks[0].srcdata)
        h_item_dst = self.proj(blocks[-1].dstdata)
        return h_item_dst + self.sage(blocks, h_item)
        
def load_model(data_dict, device, args):
    gnn = PinSAGEModel(data_dict['graph'], data_dict['item_ntype'], data_dict['textset'], args.hidden_dims, args.num_layers).to(device)
    opt = torch.optim.Adam(gnn.parameters(), lr=args.lr)
    if args.retrain:
        checkpoint = torch.load(args.save_path + '.pt', map_location=device)
    else:
        checkpoint = torch.load(args.save_path, map_location=device)
   
    gnn.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])

    return gnn, opt, checkpoint['epoch']

# 뭐 일단 요약하자면 데이터 전처리 작업.
def prepare_dataset(data_dict, args):
    g = data_dict['graph']
    item_texts = data_dict['item_texts']
    user_ntype = data_dict['user_ntype'] # user
    item_ntype = data_dict['item_ntype'] # wine

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
            #field.build_vocab(getattr(textset, key), vectors='fasttext.simple.300d')
        data_dict['textset'] = textset

    return data_dict

def prepare_dataloader(data_dict, args):
    g = data_dict['graph']
    user_ntype = data_dict['user_ntype'] # user
    item_ntype = data_dict['item_ntype'] # wine
    textset = data_dict['textset']

    # Sampler

    # 이 자슥은, 그래프, 유저 타입, 아이템 타입, 배치 사이즈를 넣고 배치 샘플링을 얻어낸다.
    batch_sampler = sampler_module.ItemToItemBatchSampler(
        g, user_ntype, item_ntype, args.batch_size)
    
    # 이 자슥은, 이웃 샘플러네 이름부터.
    neighbor_sampler = sampler_module.NeighborSampler(
        g, user_ntype, item_ntype, args.random_walk_length,
        args.random_walk_restart_prob, args.num_random_walks, args.num_neighbors,
        args.num_layers)

    # 이 자슥은, 샘플링 할때 사용하는 함수인갑다. 싶은거지.
    collator = sampler_module.PinSAGECollator(neighbor_sampler, g, item_ntype, textset)

    dataloader = DataLoader(
        batch_sampler,
        collate_fn=collator.collate_train,
        num_workers=args.num_workers)

    dataloader_test = DataLoader(
        torch.arange(g.number_of_nodes(item_ntype)),
        batch_size=args.batch_size,
        collate_fn=collator.collate_test,
        num_workers=args.num_workers)

    dataloader_it = iter(dataloader)

    return dataloader_it, dataloader_test, neighbor_sampler
    
def train(data_dict, args):
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print('Current using CPUs')
    else:
        print ('Current cuda device ', torch.cuda.current_device()) # check

    # Dataset
    data_dict = prepare_dataset(data_dict, args)
    dataloader_it, dataloader_test, neighbor_sampler = prepare_dataloader(data_dict, args)

    # Model

    # 재학습인 경우
    if args.retrain:
        print('Loading pretrained model...')
        gnn, opt, start_epoch = load_model(data_dict, device, args)

    # 일반적인 학습 상황
    else:
        # 모델 객체 생성
        gnn = PinSAGEModel(data_dict['graph'], data_dict['item_ntype'], data_dict['textset'], args.hidden_dims, args.num_layers)
        # 옵티마이저
        opt = torch.optim.Adam(gnn.parameters(), lr=args.lr)
        start_epoch = 0

    # 걍 에폭마다 평가를 할 것인지.
    if args.eval_epochs:
        g = data_dict['graph']
        item_ntype = data_dict['item_ntype']
        user_ntype = data_dict['user_ntype']
        user_to_item_etype = data_dict['user_to_item_etype']
        timestamp = data_dict['timestamp']
        nid_uid_dict = {v: k for v, k in enumerate(list(g.ndata['userID'].values())[0].numpy())}
        nid_wid_dict = {nid.item(): wid.item() for wid, nid in  zip(g.ndata['wine_id']['wine'], g.ndata['id']['wine'])}

    # 모델을 마 gpu로 보내가!
    gnn = gnn.to(device)

    # 에폭 돌리기
    for epoch in tqdm(range(start_epoch, args.num_epochs + start_epoch)):
        gnn.train()
        for batch in range(args.batches_per_epoch):
            pos_graph, neg_graph, blocks = next(dataloader_it)
            for i in range(len(blocks)):
                blocks[i] = blocks[i].to(device)
            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)

            loss = gnn(pos_graph, neg_graph, blocks).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Evaluate
        # 0번째는 재끼고
        if not epoch:
            continue

        # 1번째 에폭마다 평가
        # 조건은.. 에폭  평가할거라는 1 표시랑, 모든 에폭을 다 돌았다면..
        if args.eval_epochs and not epoch % args.eval_epochs:
            h_item = evaluation.get_all_emb(gnn, g.ndata['id'][item_ntype], data_dict['textset'], item_ntype, neighbor_sampler, args.batch_size, device)
            writelog(f"\n\nh_item: {h_item}\n\n")
            item_batch = evaluation.item_by_user_batch(g, user_ntype, item_ntype, user_to_item_etype, timestamp, args)
            writelog(f"\n\nitem_batch: {item_batch}\n\n")
            recalls = []
            precisions = []
            hitrates = []
            users = []

            for i, nodes in enumerate(item_batch):
                '''
                nodes : 유저당 실제 인터랙션 노드들 [train 노드, test 노드 (8: 2비율)]
                '''
                # 실제 유저 ID 탐색
                category = nid_uid_dict[i]
                writelog(f"\n\ncategory: {category}\n\n")
                user_id = data_dict['user_category'][category]  # 실제 유저 id
                writelog(f"\n\nuser_id: {user_id}\n\n")
                label = data_dict['testset'][user_id]  # 테스트 라벨
                writelog(f"\n\nlabel: {label}\n\n")
                users.append(user_id)

                # 실제 와인 ID 탐색
                item = evaluation.node_to_item(nodes, nid_wid_dict, data_dict['item_category'])  # 와인 ID
                writelog(f"\n\nitem: {item}\n\n")
                label_idx = [i for i, x in enumerate(item) if x in label]  # 라벨 인덱스

                # 아이템 추천
                nodes = [x for i, x in enumerate(nodes) if i not in label_idx]  # 라벨 인덱스 미포함 입력 학습용 노드
                h_nodes = h_item[nodes]
                h_center = torch.mean(h_nodes, axis=0)  # 중앙 임베딩
                dist = h_center @ h_item.t()  # 행렬곱
                topk = dist.topk(args.k)[1].cpu().numpy()  # dist 크기 순서로 k개 추출
                topk = evaluation.node_to_item(topk, nid_wid_dict, data_dict['item_category'])  # ID 변환

                tp = [x for x in label if x in topk]
                if not tp:
                    recall, precision, hitrate = 0, 0, 0
                else:
                    recall = len(tp) / len(label)
                    precision = len(tp) / len(topk)
                    hitrate = 1  # 하나라도 있음

                recalls.append(recall)
                precisions.append(precision)
                hitrates.append(hitrate)

            result_df = pd.DataFrame({'recall': recalls, 'precision': precisions, 'hitrate': hitrates})
            result_df = result_df.mean().apply(lambda x: round(x, 3))
            recall, precision, hitrate = result_df['recall'], result_df['precision'], result_df['hitrate']
            print(f'\tEpoch:{epoch}\tRecall:{recall}\tHitrate:{hitrate}\tPrecision:{precision}')

        # 에폭별 저장 여부에 따라서
        if args.save_epochs:
            if not epoch % args.save_epochs:
                torch.save({
                'epoch': epoch,
                'model_state_dict': gnn.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': loss
                        }, args.save_path + '_' + str(epoch) + 'epoch.pt')

    return gnn, epoch+1, opt, loss

if __name__ == '__main__':
    clearlog()
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset-path', type=str)
    parser.add_argument('-s', '--save-path', type=str, default='model')
    parser.add_argument('--random-walk-length', type=int, default=2)
    parser.add_argument('--random-walk-restart-prob', type=float, default=0.5)
    parser.add_argument('--num-random-walks', type=int, default=10)
    parser.add_argument('--num-neighbors', type=int, default=3)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--hidden-dims', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cpu')  # 'cpu' or 'cuda:N'
    parser.add_argument('--num-epochs', type=int, default=1)
    parser.add_argument('--batches-per-epoch', type=int, default=10000)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--eval-epochs', type=int, default=0)
    parser.add_argument('--save-epochs', type=int, default=0)
    parser.add_argument('--retrain', type=int, default=0)
    parser.add_argument('-k', type=int, default=10)
    args = parser.parse_args()

    # Load dataset
    with open(args.dataset_path, 'rb') as f:
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
    
    # Training
    gnn, epoch, opt, loss = train(data_dict, args)


    torch.save({
                'epoch': epoch,
                'model_state_dict': gnn.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': loss
            }, args.save_path + '_' + str(epoch) + 'epoch.pt')