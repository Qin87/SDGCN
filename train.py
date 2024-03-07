#############################################
# Copied and modified from DiGCN and MagNET
# https://github.com/flyingtango/DiGCN
# https://github.com/matthew-hirn/magnet
#############################################
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
import os, time, argparse, csv
from layer.sdgcn import SDGCN_Edge
from utils.edge_data_sign import generate_dataset_2class, in_out_degree, link_prediction_evaluation, load_directed_signed_graph_link, generate_dataset_2class_link, generate_dataset_2class_dir
from utils.hermitian import to_edge_dataset_sparse_sign


def parse_args():
    parser = argparse.ArgumentParser(description="SD-GCN link sign prediction")
    parser.add_argument('--log_root', type=str, default='../logs/')
    parser.add_argument('--log_path', type=str, default='test')
    parser.add_argument('--data_path', type=str, default='../dataset/data/tmp/')
    parser.add_argument('--dataset', type=str, default='BitCoinAlpha', help='data set selection')
    parser.add_argument('--split_prob', type=lambda s: [float(item) for item in s.split(',')], default="0.15,0.05")
    parser.add_argument('--epochs', type=int, default=2000, help='training epochs')
    parser.add_argument('--num_filter', type=int, default=128, help='num of filters')
    parser.add_argument('--method_name', type=str, default='SDGCN', help='method name')
    parser.add_argument('-not_norm', '-n', action='store_false')
    parser.add_argument('--q', type=float, default=0.1, help='q value for the phase matrix')
    parser.add_argument('--K', type=int, default=1, help='K for cheb series')
    parser.add_argument('--layer', type=int, default=2, help='SDGCN layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout prob')
    parser.add_argument('--num_class_link', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--l2', type=float, default=5e-4, help='l2 regularizer')
    parser.add_argument('--ensemble', type=int, default=10, help='number of ensemble model')
    parser.add_argument('--device', type=int, default=0, help='Select GPU idx')
    parser.add_argument('--ratio', type=int, default=3, help='pos_neg ratio')
    parser.add_argument('--signonly', type=int, default=0, help='ignore direction')
    parser.add_argument('--dironly', type=int, default=0, help='ignore sign')
    parser.add_argument('--linkPred', type=int, default=0, help='Direction Prediction Task')

    return parser.parse_args()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def main(args):
    data_name = args.dataset
    log_path = os.path.join(args.log_root, args.log_path, args.save_name)
    if os.path.isdir(log_path) is False:
        os.makedirs(log_path)

    dataset = load_directed_signed_graph_link(root='./dataset/data/' + data_name)

    print("Dataset Loaded " + data_name)

    if 'dataset' in locals():
        pos_edge, neg_edge = dataset
        pos_edge, neg_edge = torch.tensor(pos_edge).to(args.device), torch.tensor(neg_edge).to(args.device)
    p_max = torch.max(pos_edge).item()
    n_max = torch.max(neg_edge).item()
    size = torch.max(torch.tensor([p_max, n_max])).item() + 1

    if args.linkPred:
        datasets = generate_dataset_2class_link(pos_edge, neg_edge, splits=args.ensemble, test_prob=0.20, ratio=args.ratio)
    else:
        datasets = generate_dataset_2class(pos_edge, neg_edge, splits=args.ensemble, test_prob=0.20, ratio=args.ratio)

    results = np.zeros((args.ensemble, 2, 6))

    for i in range(args.ensemble):
        edges = datasets[i]['graph']
        pos_edges = datasets[i]['train']['pos_edge']
        neg_edges = datasets[i]['train']['neg_edge']

        if args.signonly:
            src, dst = pos_edges[:, 0], pos_edges[:, 1]
            forward = [(i, j) for i, j in zip(src, dst)]
            reverse = [(j, i) for i, j in zip(src, dst)]
            pos_edges = np.array(forward + reverse)

            src, dst = neg_edges[:, 0], neg_edges[:, 1]
            forward = [(i, j) for i, j in zip(src, dst)]
            reverse = [(j, i) for i, j in zip(src, dst)]
            neg_edges = np.array(forward + reverse)

        if args.dironly:
            pos_edges = np.array(torch.cat((torch.tensor(pos_edges), torch.tensor(neg_edges)), 0))
            neg_edges = np.array([[0, 0], [1, 1]])

        L = to_edge_dataset_sparse_sign(args.q, pos_edges, neg_edges, args.K,
                                        size, laplacian=True, norm=args.not_norm, gcn_appr=False)
        L_img, L_real = [], []
        for ind_L in range(len(L)):
            L_img.append(sparse_mx_to_torch_sparse_tensor(L[ind_L].imag).to(args.cuda))
            L_real.append(sparse_mx_to_torch_sparse_tensor(L[ind_L].real).to(args.cuda))

        X_img = in_out_degree(edges, size).to(args.cuda)
        X_real = X_img.clone()

        model = SDGCN_Edge(X_real.size(-1), L_real, L_img, K=args.K, label_dim=args.num_class_link,
                           layer=args.layer, num_filter=args.num_filter, dropout=args.dropout)
        model = model.to(args.cuda)
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

        y_train = torch.from_numpy(datasets[i]['train']['label']).long().to(args.cuda)
        y_val = torch.from_numpy(datasets[i]['validate']['label']).long().to(args.cuda)
        y_test = torch.from_numpy(datasets[i]['test']['label']).long().to(args.cuda)

        train_index = torch.from_numpy(datasets[i]['train']['pairs']).to(args.cuda)
        val_index = torch.from_numpy(datasets[i]['validate']['pairs']).to(args.cuda)
        test_index = torch.from_numpy(datasets[i]['test']['pairs']).to(args.cuda)

        #################################
        # Train/Validation/Test
        #################################
        best_test_err = 1000.0
        early_stopping = 0

        if dataset == 'BitCoinAlpha' or dataset == 'BitCoinOTC':
            early_ = 1000
        else:
            early_ = 100

        for epoch in range(args.epochs):
            if early_stopping >= early_:  # 500:
                break

            ####################
            # Train
            ####################
            model.train()
            out = model(X_real, X_img, train_index)
            train_loss = F.nll_loss(out, y_train)

            opt.zero_grad()
            train_loss.backward()
            opt.step()

            ####################
            # Validation
            ####################
            model.eval()
            out_val = model(X_real, X_img, val_index)
            val_loss = F.nll_loss(out_val, y_val)

            ####################
            # Save weights
            ####################
            save_perform = val_loss.detach().item()
            if save_perform <= best_test_err:
                early_stopping = 0
                best_test_err = save_perform
                torch.save(model.state_dict(), log_path + '/model' + str(i) + '.t7')

                ####################
                # Test
                ####################
                out_test = model(X_real, X_img, test_index)

                [[val_acc_full, val_acc, val_auc, val_f1_micro, val_f1_macro, val_f1_binary],
                 [test_acc_full, test_acc, test_auc, test_f1_micro, test_f1_macro, test_f1_binary]] = \
                    link_prediction_evaluation(out_val, out_test, y_val, y_test)
            else:
                early_stopping += 1

        results[i] = [[val_acc_full, val_acc, val_auc, val_f1_micro, val_f1_macro, val_f1_binary],
                      [test_acc_full, test_acc, test_auc, test_f1_micro, test_f1_macro, test_f1_binary]]
        log_str = ('test_acc:{test_acc:.3f}, test_auc: {test_auc:.3f}, test_f1_macro: {test_f1_macro:.3f},'
                   ' test_f1_micro: {test_f1_micro:.3f}, test_f1_binary: {test_f1_binary:.3f}')
        log_str = log_str.format(test_acc=test_acc, test_auc=test_auc, test_f1_macro=test_f1_macro,
                                 test_f1_micro=test_f1_micro, test_f1_binary=test_f1_binary)
        print('Model:' + str(i) + ' ' + log_str)

    print('Average Performance: test_acc:{:.3f}, test_auc: {:.3f}, test_f1_macro: {:.3f}, test_f1_micro: {:.3f}, test_f1_binary: {:.3f}'.format(
        np.mean(results[:, 1, 1]), np.mean(results[:, 1, 2]), np.mean(results[:, 1, 4]), np.mean(results[:, 1, 3]), np.mean(results[:, 1, 5])))
    return results


if __name__ == "__main__":
    args = parse_args()
    args.cuda = 'cuda:' + str(args.device)
    args.q = np.pi * args.q

    save_name = args.method_name + 'lr' + str(int(args.lr * 1000)) + 'num_filters' + str(
        int(args.num_filter)) + 'q' + str(int(100 * args.q)) + 'link' + str(int(args.num_class_link))
    args.save_name = save_name

    results = main(args)
