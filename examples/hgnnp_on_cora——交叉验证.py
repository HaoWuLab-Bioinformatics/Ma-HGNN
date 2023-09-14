import time
from copy import deepcopy

import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torchinfo import summary
from dhg import Graph, Hypergraph
from dhg.data import Cora, Pubmed, Citeseer, Facebook
from dhg.models import HGNN, HGNNP, HNHN, GCN, GAT

from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator
from dhg.data import Cooking200

def train(net, X, G, lbls, train_idx, optimizer):
    net.train()

    st = time.time()
    optimizer.zero_grad()
    outs = net(X, G)
    outs, lbls = outs[train_idx], lbls[train_idx]
    loss = F.cross_entropy(outs, lbls)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def infer(net, X, G, lbls, idx, test=False):
    net.eval()
    outs = net(X, G)
    outs, lbls = outs[idx], lbls[idx]
    if not test:
        res = evaluator.validate(lbls, outs)
    else:
        res = evaluator.test(lbls, outs)
    return res


if __name__ == "__main__":
    #set_seed(2023)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])
    #data = Cora()
    #data =Pubmed()
    data = Citeseer()

    X, lbl = data["features"], data["labels"]
    G = Graph(data["num_vertices"], data["edge_list"])
    HG = Hypergraph.from_graph(G)


    train_mask = data["train_mask"]
    val_mask = data["val_mask"]
    test_mask = data["test_mask"]

    net = HGNNP(data["dim_features"], 64, data["num_classes"])
    optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=0.0005)

    X, lbl = X.to(device), lbl.to(device)
    HG = HG.to(X.device)
    net = net.to(device)

    kf = KFold(n_splits=5,shuffle=True)
    best_acc = 0
    accave=0
    best_epoch, best_fold = 0, 0
    best_state = None
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"processing fold {fold_idx + 1}/5...")
        best_val = 0
        for epoch in range(100):
                # train
            loss = train(net, X, HG, lbl, train_idx, optimizer)

                # validation
            if epoch % 1 == 0:
                with torch.no_grad():
                        val_res = infer(net, X, HG, lbl, val_mask)
                if val_res > best_val:
                        print(f"fold {fold_idx + 1}, update best: {val_res:.5f}")
                        best_epoch = epoch
                        best_val = val_res
                        best_state = deepcopy(net.state_dict())

            # test
        print("testing...")
        net.load_state_dict(best_state)
        res = infer(net, X, HG, lbl, test_mask, test=True)
            #print(res)
        acc = res["accuracy"]
            #print(f"fold {fold_idx + 1}, best val: {best_val:.5f}, test result: accuracy={acc:.5f}")

        if acc > best_acc:
                best_acc = acc
                best_fold = fold_idx + 1
        accave=acc+accave
    print("accave:")
    print(accave/5)
    print(f"\ntrain finished! Best fold: {best_fold}, best accuracy: {best_acc:.5f}.")
    print("---------------------------------------------------------------------------------------------------")



