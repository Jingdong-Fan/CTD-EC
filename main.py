import numpy as np
from diffuse.diffuse import Diffuse
import os
import torch
import random
import sys
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
np.set_printoptions(precision=3)
sys.stdout = open('main.txt', 'w')

def compute_evaluate(W_p, W_true):
    assert (W_p.shape == W_true.shape and W_p.shape[0] == W_p.shape[1])
    TP = np.sum((W_p + W_true) == 2)
    TP_FP = W_p.sum(axis=1).sum()
    TP_FN = W_true.sum(axis=1).sum()
    TN = ((W_p + W_true) == 0).sum()

    accuracy = (TP + TN) / (W_p.shape[0]*W_p.shape[0])
    precision = TP / TP_FP
    recall = TP / TP_FN
    F1 = 2 * (recall * precision) / (recall + precision)
    shd = np.count_nonzero(W_p != W_true)

    mt = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'F1': F1,'shd': shd}
    for i in mt:
        mt[i] = round(mt[i], 4)
    return mt

def main():
    epochs = int(4000)
    batch_size = 200

    n_steps = int(200)
    beta_start = 0.0001
    beta_end = 0.02

    small_layer = 64

    subject_num = 50
    subject_length = 200

    X = np.loadtxt(file_path)
    true_causal_matrix = np.loadtxt(gtrue_path)
    n_nodes = true_causal_matrix.shape[0]
    adj_all = np.zeros((subject_num, n_nodes, n_nodes))

    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    diffuse = Diffuse(small_layer,  n_nodes,  beta_start, beta_end,
                     epochs, batch_size, n_steps)

    diffuse.fit1(X,condition)

    for id_subject in range(subject_num):
        start = int(id_subject * (subject_length))
        end = int((id_subject + 1) * (subject_length))
        prunX = X[start:end, :]
        sub_adj_matrix = diffuse.fit2(prunX)

        for i_subject in range(n_nodes):
            for j_subject in range(n_nodes):
                if sub_adj_matrix[i_subject][j_subject] == 1:
                    adj_all[id_subject][i_subject][j_subject] = 1

    precision_all = []
    recall_all = []
    f1_all = []
    shd_all = []

    for id_subject in range(subject_num):
        mt = compute_evaluate(adj_all[id_subject], true_causal_matrix)

        precision_all.append(mt['precision'])
        recall_all.append(mt['recall'])
        f1_all.append(mt['F1'])
        shd_all.append(mt['shd'])

    mean_precision = np.mean(precision_all)
    std_precision = np.std(precision_all)
    mean_recall = np.mean(recall_all)
    std_recall = np.std(recall_all)
    mean_F1 = np.mean(f1_all)
    std_F1 = np.std(f1_all)
    mean_shd = np.mean(shd_all)
    std_shd = np.std(shd_all)
    print(file_path)
    print("mean+std--precision: {:.2f} + {:.2f}".format( mean_precision, std_precision))
    print("mean+std--recall: {:.2f} + {:.2f}".format( mean_recall, std_recall))
    print("mean+std--F1: {:.2f} + {:.2f}".format( mean_F1, std_F1))
    print("mean+std--shd: {:.2f} + {:.2f}".format( mean_shd, std_shd))


if __name__ == "__main__":

        file_path = 'simsTxt/sim1.txt'
        gtrue_path = 'simsTxt/stand_5nodes.txt'
        condition = np.loadtxt("trans/encoderout1.txt")
        main()
        file_path = 'simsTxt/sim2.txt'
        gtrue_path = 'simsTxt/stand_5nodes.txt'
        condition = np.loadtxt("trans/encoderout2.txt")
        main()
        file_path = 'simsTxt/sim3.txt'
        gtrue_path = 'simsTxt/stand_5nodes.txt'
        condition = np.loadtxt("trans/encoderout3.txt")
        main()
        file_path = 'simsTxt/sim4.txt'
        gtrue_path = 'simsTxt/stand_5nodes.txt'
        condition = np.loadtxt("trans/encoderout4.txt")
        main()
        file_path = 'simsTxt/sim5.txt'
        gtrue_path = 'simsTxt/stand_10nodes.txt'
        condition = np.loadtxt("trans/encoderout5.txt")
        main()

