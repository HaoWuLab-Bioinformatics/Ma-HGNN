import csv
import os
import time
import copy
import torch
import torch.optim as optim
import pprint as pp
from dhg.nn.convs.hypergraphs import hypergraph_utils as hgut
from config import get_config
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error,f1_score,roc_auc_score,r2_score
from sklearn.metrics import f1_score
import warnings
from dhg import Graph, Hypergraph
from dhg.models import HGNN, HGNNP, HNHN
import matplotlib.pyplot as plt


warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
def load_st_construct_H(
                             m_prob=1,
                             K_neigs=[1],
                             #K_neigs=3,
                             is_probH=True,
                             split_diff_scale=False,
                             use_st_feature=True,
                             use_st_feature_for_structure=True,
                            ):

    if True:
        student_feature = pd.read_csv('data.csv', encoding='ISO-8859-1')

        fts = student_feature.iloc[0:500, 0:7].astype(np.float32).values  # 特征
        lbls = student_feature.iloc[0:500, 7].values  # 标签
        fts_, x_val, lbls_, y_val = train_test_split(fts, lbls, train_size=0.9, random_state=0)
        fts_area = fts_[:, 0:3].astype(np.float32)  # 地区
        fts_goods = fts_[:, 3:4].astype(np.float32)  # 商品
        fts_price = fts_[:, 4:6].astype(np.float32)  # 价格
        fts_time = fts_[:, 6:7].astype(np.float32)  # 时间


        #fts, lbls = RandomOverSampler().fit_resample(fts, lbls)
        x = pd.DataFrame(fts_)
        y = pd.Series(lbls_)
        x_train, x_val, y_train, y_val = train_test_split(x,y, train_size = 0.9, random_state = 0)
        idx_train = x_train.index
        idx_test = x_val.index
    # construct feature matrix
    # construct hypergraph incidence matrix
    print('Constructing hypergraph incidence matrix! \n(It may take several minutes! Please wait patiently!)')
    H = None

    if use_st_feature_for_structure:
        tmp = hgut.construct_muiH_with_KNN(fts_area,fts_goods,fts_price,fts_time, K_neigs=K_neigs,
                                        split_diff_scale=split_diff_scale,
                                        is_probH=is_probH, m_prob=m_prob)
        print(tmp.shape)
        H = hgut.hyperedge_concat(H, tmp)
    if H is None:
        raise Exception('None feature to construct hypergraph incidence matrix!')

    return fts_,  lbls_, idx_train, idx_test, H


fts, lbls, idx_train, idx_test, H = \
    load_st_construct_H(
                             m_prob=1,
                             K_neigs=[2],
                             is_probH=True,
                             split_diff_scale=False,
                             use_st_feature=False,
                             use_st_feature_for_structure=True,
                            )

H=torch.tensor(H)
G = Hypergraph.from_feature_kNN(H,2)


n_class = int(lbls.max()) + 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# transform data to device
fts = torch.Tensor(fts).to(device)
lbls = torch.Tensor(lbls).squeeze().long().to(device)
#G = torch.Tensor(G).to(device)
idx_train = torch.Tensor(idx_train).long().to(device)
idx_test = torch.Tensor(idx_test).long().to(device)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25, print_freq=500):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    meanlist=[]
    acclist=[]
    f1scorelist=[]
    r2scorelist=[]
    presionlist=[]
    xlist=[]
    for epoch in range(1000):
        if epoch % print_freq == 0:
            print('-' * 10)
            print(f'Epoch {epoch}/{num_epochs - 1}')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            idx = idx_train if phase == 'train' else idx_test

            # Iterate over data.  outputs[idx]:预测值，lbls[idx]：实际值
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(fts, G)
                
                loss = criterion(outputs[idx], lbls[idx])
                _, preds = torch.max(outputs, 1)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            #print(preds)
            # statistics
            running_loss += loss.item() * fts.size(0)
            running_corrects += torch.sum(preds[idx] == lbls.data[idx])

            epoch_loss = running_loss / len(idx)
            epoch_acc = running_corrects.double() / len(idx)
            accuracy = accuracy_score(y_true=lbls[idx].cpu().numpy(), y_pred=preds[idx].cpu().numpy())
            classification = classification_report(y_true=lbls[idx].cpu().numpy(), y_pred=preds[idx].cpu().numpy())
            mean=mean_squared_error(lbls[idx].cpu().numpy(),preds[idx].cpu().numpy())
            mean=np.sqrt(mean)
            r2score=r2_score(lbls[idx].cpu().numpy(),preds[idx].cpu().numpy())
            #f1score=f1_score(lbls[idx].cpu().numpy(),preds[idx].cpu().numpy())
            acc=accuracy
            presion=precision_score(lbls[idx].cpu().numpy(), preds[idx].cpu().numpy(),average='macro')
            f1score=f1_score(lbls[idx].cpu().numpy(), preds[idx].cpu().numpy(),average='weighted')
            if epoch % print_freq == 0:
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}' )
                print(accuracy)
                print(classification)
                print(mean)
                print(fts.shape)
            if phase=='train':
                meanlist.append(mean)
                r2scorelist.append(r2score)
                acclist.append(acc)
                presionlist.append(presion)
                f1scorelist.append(f1score)

            # deep copy the model
            if phase == 'val' and accuracy > best_acc:
                best_acc = accuracy
                best_model_wts = copy.deepcopy(model.state_dict())

        if epoch % print_freq == 0:
            print(f'Best val Acc: {best_acc:4f}')
            print('-' * 20)

        xlist.append(epoch)

    time_elapsed = time.time() - since
    np.savetxt('x_pos.txt',np.array(outputs.cpu().detach().numpy()  ))
    np.savetxt('y_pos.txt',np.array(lbls.cpu().detach().numpy()))
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)

    with open ('acctest-copy_fin.csv','w') as f:
        writer=csv.writer(f)
        writer.writerow(xlist)
        writer.writerow(acclist)


    with open ('r2scoretest-copy_fin.csv','w') as m:
        writer=csv.writer(m)
        writer.writerow(xlist)
        writer.writerow(r2scorelist)

    with open ('f1scoretest-copy_fin.csv','w') as a:
        writer=csv.writer(a)
        writer.writerow(xlist)
        writer.writerow(f1scorelist)

    with open ('meantest-copy_fin.csv','w') as b:
        writer=csv.writer(b)
        writer.writerow(xlist)
        writer.writerow(meanlist)


    with open ('presiontest-copy_fin.csv','w') as c:
        writer=csv.writer(c)
        writer.writerow(xlist)
        writer.writerow(presionlist)


    plt.plot(xlist, acclist)
    plt.show()

    return model


def _main():


    model_ft = HGNNP(in_channels=fts.shape[1],hid_channels=128,num_classes=n_class,drop_rate=0.5)
    model_ft = model_ft.to(device)

    optimizer = optim.Adam(model_ft.parameters(), lr=0.01,
                           weight_decay=0.0001)

    schedular = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=[100],
                                               gamma=0.8)
    
    criterion = torch.nn.CrossEntropyLoss()
    model_ft = train_model(model_ft, criterion, optimizer, schedular, 10000, print_freq=50)
    #torch.save(model_ft.state_dict(), "modelfin.pth")


if __name__ == '__main__':
    _main()
