from torch import nn, optim
from torch.utils.data import TensorDataset
from src import *
from src.ftl_net import FTL_nn
from src.distribute_data import Distribute_Data
from sklearn import metrics
import time
import copy
import warnings
warnings.filterwarnings("ignore")
import torch

def wasserstein_distance(x, y, epsilon=0.5, n_iters=1):
    device = x.device

    # 计算pairwise距离矩阵
    C = torch.cdist(x.unsqueeze(0), y.unsqueeze(0), p=2).squeeze(0)  # [n, m]

    # 均匀分布权重
    n, m = C.shape
    mu = torch.full((n,), 1.0 / n, device=device)
    nu = torch.full((m,), 1.0 / m, device=device)

    # 初始化对偶变量
    u = torch.zeros_like(mu)
    v = torch.zeros_like(nu)

    for _ in range(n_iters):
        u = epsilon * (torch.log(mu + 1e-8) - torch.logsumexp((v.unsqueeze(0) - C) / epsilon, dim=1)) + u
        v = epsilon * (torch.log(nu + 1e-8) - torch.logsumexp((u.unsqueeze(1) - C) / epsilon, dim=0)) + v

    # 计算传输矩阵
    pi = torch.exp((u.unsqueeze(1) + v.unsqueeze(0) - C) / epsilon)
    dist = (pi * C).sum()
    return dist

def cal_acc(model, dataloader, dataset_name, alignment):

    AUC_PR = []
    AUC_ROC = []
    BS_Plus = []
    KS = []
    F1 = []
    with torch.no_grad():
        for data_ptr, label in dataloader:
            outputs, fea_outputs = model.forward(data_ptr, alignment)
            if alignment == 'aligned':
                outputs = outputs[0] + outputs[1] + outputs[2]
            elif alignment == 'non_aligned':
                outputs = outputs[0]
            output_prob = nn.functional.softmax(outputs, dim=-1)[:, 1]
            pred = outputs.max(1, keepdim=True)[1]

            confusion = metrics.confusion_matrix(label.cpu().numpy(), pred.detach().numpy())
            FN = confusion[1][0]
            TN = confusion[0][0]
            TP = confusion[1][1]
            FP = confusion[0][1]


            AUC_ROC.append(metrics.roc_auc_score(label, output_prob))
            precision, recall, _ = metrics.precision_recall_curve(label, output_prob)
            auc_pr = metrics.auc(recall, precision)
            AUC_PR.append(auc_pr)

            bs_plus = metrics.brier_score_loss(label.detach().numpy(), output_prob.detach().numpy())
            BS_Plus.append(bs_plus)
            fpr, tpr, thresholds = metrics.roc_curve(label, output_prob)
            ks = max(abs(fpr - tpr))
            KS.append(ks)




    print("{}: AUC-ROC {:.4f}".format(dataset_name, np.mean(AUC_ROC)))
    print("{}: AUC-PR {:.4f}".format(dataset_name, np.mean(AUC_PR)))
    print("{}: KS {:.4f}".format(dataset_name, np.mean(KS)))
    print("{}: BS+ {:.4f}".format(dataset_name, np.mean(BS_Plus)))

    return np.round(np.mean(AUC_ROC), 4), np.round(np.mean(AUC_PR), 4), np.round(np.mean(BS_Plus), 4), np.round(
        np.mean(KS), 4)

def post_data(data_fea, data_labels,batch):
    X_train, X_test, Y_train, Y_test = train_test_split(data_fea, data_labels, test_size=0.2, random_state=11)
    aligned_X_train, non_aligned_X_train, aligned_Y_train, non_aligned_Y_train = train_test_split(X_train, Y_train,
                                                                                                  test_size=0.8,
                                                                                                  random_state=11)
    aligned_X_test, non_aligned_X_test, aligned_Y_test, non_aligned_Y_test = train_test_split(X_test, Y_test,
                                                                                                 test_size=0.8,
                                                                                                 random_state=11)
    print('aligned_X_train',aligned_X_train.shape)
    print('non_aligned_X_train',non_aligned_X_train.shape)
    print('aligned_X_test',aligned_X_test.shape)
    print('non_aligned_X_tes',non_aligned_X_test.shape)
    # 分别封装对齐训练、测试样本和非对齐训练、测试样本
    aligned_train_fea = torch.tensor(aligned_X_train.astype(float).values, dtype=torch.float)
    aligned_train_labels = torch.tensor(aligned_Y_train.values, dtype=torch.long)
    aligned_dataset_train = TensorDataset(aligned_train_fea, aligned_train_labels)
    aligned_train_loader = torch.utils.data.DataLoader(aligned_dataset_train, batch_size=batch, shuffle=True)

    non_aligned_train_fea = torch.tensor(non_aligned_X_train.astype(float).values, dtype=torch.float)
    non_aligned_train_labels = torch.tensor(non_aligned_Y_train.values, dtype=torch.long)
    non_aligned_dataset_train = TensorDataset(non_aligned_train_fea, non_aligned_train_labels)
    non_aligned_train_loader = torch.utils.data.DataLoader(non_aligned_dataset_train, batch_size=600000, shuffle=True)

    aligned_test_fea = torch.tensor(aligned_X_test.astype(float).values, dtype=torch.float)
    aligned_test_labels = torch.tensor(aligned_Y_test.values, dtype=torch.long)
    aligned_dataset_test = TensorDataset(aligned_test_fea, aligned_test_labels)
    aligned_test_loader = torch.utils.data.DataLoader(aligned_dataset_test, batch_size=600000, shuffle=True)

    non_aligned_test_fea = torch.tensor(non_aligned_X_test.astype(float).values, dtype=torch.float)
    non_aligned_test_labels = torch.tensor(non_aligned_Y_test.values, dtype=torch.long)
    non_aligned_dataset_test = TensorDataset(non_aligned_test_fea, non_aligned_test_labels)
    non_aligned_test_loader = torch.utils.data.DataLoader(non_aligned_dataset_test, batch_size=600000, shuffle=True)

    return aligned_train_loader, aligned_test_loader, non_aligned_train_loader, non_aligned_test_loader

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def fea_distill(fea_stu, fea_tea):
    loss_fea = nn.functional.kl_div(fea_stu, fea_tea.detach(), reduction='mean')
    return loss_fea
def fea_train(x, target, ftl_nn_fa,ftl_nn_fn, alignment, ss):
    ftl_nn_fn.zero_grads()
    ftl_nn_fn_outputs, ftl_nn_fn_fea_outputs = ftl_nn_fn.forward(x, 'aligned')
    ftl_nn_fa_outputs, ftl_nn_fa_fea_outputs = ftl_nn_fa.forward(x, 'aligned')
    if ss == 'f':
        # 计算损失函数
        loss1 = wasserstein_distance(ftl_nn_fa_fea_outputs[0].detach(), ftl_nn_fn_fea_outputs[0])
        loss2 = wasserstein_distance(ftl_nn_fa_fea_outputs[1].detach(), ftl_nn_fn_fea_outputs[1])
        loss3 = wasserstein_distance(ftl_nn_fa_fea_outputs[2].detach(), ftl_nn_fn_fea_outputs[2])
        loss1.backward()
        loss2.backward()
        loss3.backward()
    else:

        # 非对称损失函数增强不平衡学习性能
        pt = nn.functional.softmax(ftl_nn_fn_outputs[0], dim=-1)[:, 1]
        # 在原始ce上增加动态权重因子
        λ = 5
        a = 2
        x = target[:ftl_nn_fn_outputs[0].size(0)] - pt
        loss1 = torch.mean(λ * (1 - 1 / (1 + (torch.exp(a * x) - (a * x) - 1) * abs(x))))

        loss1.backward()
    ftl_nn_fn.step()
    return loss1.detach().item()
def contrastive_train(x, target, splitNN, alignment, ss):
    # 1) Zero our grads
    splitNN.zero_grads()
    # 2) Make a prediction
    outputs, fea_outputs = splitNN.forward(x, alignment)
    if ss == 'f':
        '跟target没关系，和fea_outputs有关系'
        all_fea = (fea_outputs[0] + fea_outputs[1] + fea_outputs[2])


        loss1 = wasserstein_distance(all_fea.detach(), fea_outputs[0])

        loss2 = wasserstein_distance(all_fea.detach(), fea_outputs[1])

        loss3 = wasserstein_distance(all_fea.detach(), fea_outputs[2])
        loss1.backward()
        loss2.backward()
        loss3.backward()
    else:

        # 非对称损失函数增强不平衡学习性能
        pt1 = nn.functional.softmax(outputs[0], dim=-1)[:, 1]
        pt2 = nn.functional.softmax(outputs[1], dim=-1)[:, 1]
        pt3 = nn.functional.softmax(outputs[2], dim=-1)[:, 1]
        x1 = target - pt1
        x2 = target - pt2
        x3 = target - pt3

        λ = 5
        a = 2
        loss1 = torch.mean(λ*(1 - 1 / (1 + (torch.exp(a * x1) - (a * x1) - 1) * abs(x1))))
        loss2 = torch.mean(λ*(1 - 1 / (1 + (torch.exp(a * x2) - (a * x2) - 1) * abs(x2))))
        loss3 = torch.mean(λ*(1 - 1 / (1 + (torch.exp(a * x3) - (a * x3) - 1) * abs(x3))))

        loss1.backward()
        loss2.backward()
        loss3.backward()
    splitNN.step()
    loss = (loss1 + loss2 + loss3) / 3
    return loss.detach().item()

if __name__ == "__main__":
    # 定义随机种子
    torch.manual_seed(11)
    # load credit data
    dataset_name = "HMEQ"
    data_fea, data_labels, X_train, X_test, Y_train, Y_test = load_data(dataset_name)
    pre_epochs = 1
    epochs = 50
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    data_owners = ['client_1', 'client_2', 'client_3']
    model_locations = ['client_1', 'client_2', 'client_3']
    if dataset_name == "Taiwan":
        batch = 256
        input_size = [16, 33, 42]
        hidden_sizes = {"client_1": [14, 6], "client_2": [14, 6], "client_3": [14, 6]}

    elif dataset_name == "GMSC":
        batch = 1280
        input_size = [2, 4, 4]
        hidden_sizes = {"client_1": [4, 4], "client_2": [4, 4], "client_3": [4, 4]}
    elif dataset_name == "LD":
        batch = 32
        input_size = [4, 8, 16]
        hidden_sizes = {"client_1": [8, 6], "client_2": [8, 6], "client_3": [8, 6]}

    elif dataset_name == "German":
        batch = 32
        input_size = [7, 25, 29]
        hidden_sizes = {"client_1": [12, 8], "client_2": [12, 8], "client_3": [12, 8]}

    elif dataset_name == "HMEQ":
        batch = 64
        input_size = [14, 25, 16]
        hidden_sizes = {"client_1": [12, 8], "client_2": [12, 8], "client_3": [12, 8]}

    elif dataset_name == "LC":
        batch = 512
        input_size = [7, 12, 15]
        hidden_sizes = {"client_1": [8, 6], "client_2": [8, 6], "client_3": [8, 6]}
    elif dataset_name == "PAKDD":
        batch = 512
        input_size = [4, 7, 9]
        hidden_sizes = {"client_1": [6, 4], "client_2": [6, 4], "client_3": [6, 4]}
    elif dataset_name == "HC":
        batch = 1280
        input_size = [79, 120, 180]
        hidden_sizes = {"client_1": [128, 64], "client_2": [128, 64], "client_3": [128, 64]}
    elif dataset_name == "Ant":
        batch = 256
        input_size = [9, 10, 10]
        hidden_sizes = {"client_1": [8, 6], "client_2": [8, 6], "client_3": [8, 6]}

    # 划分对齐样本和非对齐样本至多个参与方
    aligned_train_loader, aligned_test_loader, non_aligned_train_loader, non_aligned_test_loader = post_data(data_fea,
                                                                                                             data_labels,
                                                                                                             batch)
    aligned_distributed_trainloader = Distribute_Data(data_owners=data_owners, data_loader=aligned_train_loader,
                                                      data_name=dataset_name,alignment='aligned',data_from='train')
    non_aligned_distributed_trainloader = Distribute_Data(data_owners=data_owners, data_loader=non_aligned_train_loader,
                                                      data_name=dataset_name,alignment='non_aligned',data_from='train')
    aligned_distributed_testloader = Distribute_Data(data_owners=data_owners, data_loader=aligned_test_loader,
                                                     data_name=dataset_name,alignment='aligned',data_from='test')
    non_aligned_distributed_testloader = Distribute_Data(data_owners=data_owners, data_loader=non_aligned_test_loader,
                                                     data_name=dataset_name,alignment='non_aligned',data_from='test')


    class client_model(nn.Module):
        def __init__(self, input_sizes, hidden_sizes, output_sizes):
            super(client_model, self).__init__()
            self.fc1 = nn.Linear(input_sizes, hidden_sizes[0])
            self.relu1 = torch.nn.ReLU()
            self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
            self.relu2 = torch.nn.ReLU()
            self.fc3 = nn.Linear(hidden_sizes[1], output_sizes)

        def forward(self, x):
            x = self.relu1(self.fc1(x))
            x = self.relu2(self.fc2(x))
            feature1 = x
            x = self.fc3(x)
            return x, feature1
    '----------------------------共享样本纵向联邦聚合----------------------------'
    print('----------------------------预训练fa模型特征提取层网络（对比学习） -----------------------------')
    fa_models = {
        "client_1": client_model(input_size[0], hidden_sizes["client_1"], 2),
        "client_2": client_model(input_size[1], hidden_sizes["client_2"], 2),
        "client_3": client_model(input_size[2], hidden_sizes["client_3"], 2), }
    # Create optimisers for each segment and link to their segment
    optimizers = [optim.SGD(fa_models[location].parameters(), lr=0.01) for location in model_locations]
    ftl_nn_fa = FTL_nn(fa_models, optimizers, data_owners).to(device)
    AUC_PR_epoch = []
    AUC_ROC_epoch = []
    BS_Plus_epoch = []
    KS_epoch = []
    F1_epoch = []
    start_time = time.time()
    for i in range(pre_epochs):
        running_loss = 0
        ftl_nn_fa.train()
        for images, labels in aligned_distributed_trainloader:
            # 对比学习训练
            loss = contrastive_train(images, labels, ftl_nn_fa, 'aligned', 'f')
            running_loss += loss
        # else:
        #     print("Epoch {} - Training loss: {}".format(i, running_loss / len(aligned_train_loader)))
    print('fa预训练完成')
    print('----------------------------基于预训练后的fa模型进行完整训练（不冻结fa特征提取网络层参数）----------------------------')

    for i in range(epochs):
        running_loss = 0
        ftl_nn_fa.train()
        for images, labels in aligned_distributed_trainloader:
            # 对比学习训练
            loss = contrastive_train(images, labels, ftl_nn_fa, 'aligned', 'p')
            running_loss += loss
        else:
            print("Epoch {} - Training loss: {}".format(i, running_loss / len(non_aligned_train_loader)))
    # 保存训练完成后的aligned样本特征提取器
    torch.save(ftl_nn_fa.models['client_1'].state_dict(), './model_param/client_1.pkl')
    torch.save(ftl_nn_fa.models['client_2'].state_dict(), './model_param/client_2.pkl')
    torch.save(ftl_nn_fa.models['client_3'].state_dict(), './model_param/client_3.pkl')
    end_time = time.time()
    print('{:.4f} s'.format(end_time - start_time))
    print('对齐测试样本的性能：')
    auc_roc_epoch_1, auc_pr_epoch_1, bs_plus_1, ks_epoch_1= cal_acc(ftl_nn_fa,aligned_distributed_testloader,
                                                                                "Test set", 'aligned')

    '----------------------------非共享样本横向联邦聚合 - ---------------------------'
    print('----------------------------预训练fn模型特征提取层网络（特征蒸馏+fc2层横向聚合） ----------------------------')
    # Create optimisers for each segment and link to their segment
    fn_models = {
        "client_1": client_model(input_size[0], hidden_sizes["client_1"], 2),
        "client_2": client_model(input_size[1], hidden_sizes["client_2"], 2),
        "client_3": client_model(input_size[2], hidden_sizes["client_3"], 2), }
    optimizers = [optim.SGD(fn_models[location].parameters(), lr=0.01,) for location in model_locations]
    ftl_nn_fn = FTL_nn(fn_models, optimizers, data_owners).to(device)
    for i in range(pre_epochs):
        running_loss = 0
        ftl_nn_fn.train()
        for images, labels in non_aligned_distributed_trainloader:
            # 特征蒸馏学习训练
            loss = fea_train(images, labels, ftl_nn_fa, ftl_nn_fn, 'non_aligned','f')
            running_loss += np.mean(loss)


        # fn训练完成后进行部分层参数共享和横向聚合
        model1 = ftl_nn_fn.models['client_1'].state_dict()
        model2 = ftl_nn_fn.models['client_2'].state_dict()
        model3 = ftl_nn_fn.models['client_3'].state_dict()
        agg_nn_weights = [model1, model2, model3]
        conv_dict = ['fc2.weight', 'fc2.bias']
        pretrained_dict1 = {k: v for k, v in agg_nn_weights[0].items() if k in conv_dict}
        pretrained_dict2 = {k: v for k, v in agg_nn_weights[1].items() if k in conv_dict}
        pretrained_dict3 = {k: v for k, v in agg_nn_weights[2].items() if k in conv_dict}
        agg_nn_w = [pretrained_dict1, pretrained_dict2, pretrained_dict3]
        agg_nn_glob = FedAvg(agg_nn_w)
        model1.update(agg_nn_glob)
        ftl_nn_fn.models['client_1'].load_state_dict(model1)
        model2.update(agg_nn_glob)
        ftl_nn_fn.models['client_2'].load_state_dict(model2)
        model3.update(agg_nn_glob)
        ftl_nn_fn.models['client_3'].load_state_dict(model3)
    print('fn模型预训练完成')
    print('----------------------------基于预训练后的fn模型进行完整训练----------------------------')
    # 冻结模型特征提取网络层参数，只更新最后的全连接网络层参数
    optimizers = [optim.SGD(fn_models[location].parameters(), lr=0.01, ) for location in model_locations]
    # optimizers = [optim.Adam(fn_models[location].parameters(), lr=0.01, ) for location in model_locations]
    ftl_nn_fn = FTL_nn(fn_models, optimizers, data_owners).to(device)
    for i in range(epochs):
        running_loss = 0
        ftl_nn_fn.train()
        for images, labels in non_aligned_distributed_trainloader:
            # 特征蒸馏学习训练
            loss = fea_train(images, labels, ftl_nn_fa, ftl_nn_fn, 'non_aligned', 'p')
            running_loss += np.mean(loss)
        else:
            print("Epoch {} - Training loss: {}".format(i, running_loss / len(non_aligned_train_loader)))
    print('非对齐测试样本的性能：')
    auc_roc_epoch_2, auc_pr_epoch_2, bs_plus_2, ks_epoch_2 = cal_acc(ftl_nn_fn,
                                                                                 non_aligned_distributed_testloader,
                                                                                 "Test set", 'non_aligned')
    print('测试样本的总体性能：')
    print('Test set: AUC-ROC', (auc_roc_epoch_1 + auc_roc_epoch_2) / 2)
    print('Test set: AUC-PR', (auc_pr_epoch_1 + auc_pr_epoch_2) / 2)
    print('Test set: KS', (ks_epoch_1 + ks_epoch_2) / 2)

    print('Test set: BS+', (bs_plus_1 + bs_plus_2) / 2)
    print(np.round((auc_roc_epoch_1 + auc_roc_epoch_2) / 2, 4))
    print(np.round((auc_pr_epoch_1 + auc_pr_epoch_2) / 2, 4))
    print(np.round((ks_epoch_1 + ks_epoch_2) / 2, 4))
    print(np.round((bs_plus_1 + bs_plus_2) / 2, 4))

