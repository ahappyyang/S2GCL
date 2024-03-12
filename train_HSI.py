from time import perf_counter as t
import random
import time
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from scipy.sparse import coo_matrix, csr_matrix
from data_unit.utils import blind_other_gpus, row_normalize, sparse_mx_to_torch_sparse_tensor,normalize_graph
from models import LogReg, S2GCL_Fast
from torch_geometric.utils import degree
import os
import argparse
from sklearn.cluster import KMeans
from ruamel.yaml import YAML
from termcolor import cprint
from evaluate import mask_test_edges
import LDA_SLIC
from functions import get_data,normalize,data_process,spixel_to_pixel_labels,cluster_accuracy,get_args,color_results,pprint_args,get_dataset
import imageio
import seaborn as sns


# ===================================================#
seed=9
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#选择cpu或者GPU
# ===================================================#




def run_S2GCL(args, dataset,gt_reshape,Q, dataset_name, gt):

    # ===================================================#
    data, adj_list, x_list, nb_list = get_dataset(dataset)
    lable = nb_list[0]#标签
    nb_feature = nb_list[1]#特征维度
    nb_classes = nb_list[2]#类别数
    nb_nodes = nb_list[3]#超像素个数
    feature_X = x_list[0].to(device)#特征值


    Y=feature_X.t()
    a=torch.cdist(Y,Y,p=2)
    a = -torch.square(a)/0.2
    a = torch.exp(a)

    k_nn=2
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if abs(j-i)>=k_nn:
                a[i][j]=0


    A_I_nomal = adj_list[0].to(device)#归一化邻接矩阵
    ylabelsx = []
    for i in range(0, nb_nodes):
        ylabelsx.append(i)
    ylablesx = np.array(ylabelsx)
    adj_1 = adj_list[1]

    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
    test_edges, test_edges_false = mask_test_edges(adj_1, test_frac=0.1, val_frac=0.05)#很有效果
    cprint("## Done ##", "yellow")
    # ===================================================#
    model = S2GCL_Fast(nb_feature, cfg=args.cfg,
                       dropout=args.dropout)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.to(device)

    A_degree = degree(A_I_nomal._indices()[0], nb_nodes, dtype=int).tolist()#节点的度
    edge_index = A_I_nomal._indices()[1]
    # ===================================================#
    my_margin = args.margin1
    my_margin_2 = my_margin + args.margin2
    margin_loss = torch.nn.MarginRankingLoss(margin=my_margin, reduce=False)
    num_neg = args.NN
    lbl_z = torch.tensor([0.]).to(device)
    deg_list_2 = []
    deg_list_2.append(0)
    for i in range(nb_nodes):
        deg_list_2.append(deg_list_2[-1] + A_degree[i])
    idx_p_list = []
    for m in range(5):
        for j in range(1, 101):
            random_list = [deg_list_2[i] + j % A_degree[i] for i in range(nb_nodes)]
            idx_p = edge_index[random_list]
            idx_p_list.append(idx_p)
        for current_iter, epoch in enumerate(tqdm(range(args.start_epoch, args.start_epoch + args.epochs + 1))):
            model.train()
            optimiser.zero_grad()
            idx_list = []
            for i in range(num_neg):
                idx_0 = np.random.permutation(nb_nodes)
                idx_list.append(idx_0)

            h_a, h_p = model(feature_X,a, A_I_nomal)#h_a是经过MLP的输出 h_p是经过MLP再乘以邻接矩阵    h_a-->h(锚嵌入)   h_p-->h+ (结构嵌入)

            h_p_1 = (h_a[idx_p_list[epoch % 100]] + h_a[idx_p_list[(epoch + 2) % 100]] + h_a[
                idx_p_list[(epoch + 4) % 100]] + h_a[idx_p_list[(epoch + 6) % 100]] + h_a[
                         idx_p_list[(epoch + 8) % 100]]) / 5 #h_p_1-->h+~ (邻居嵌入)
            s_p = F.pairwise_distance(h_a, h_p)#s_p-->d(h,h+)
            s_p_1 = F.pairwise_distance(h_a, h_p_1)#s_p_1-->d(h,h+~)
            s_n_list = []
            for h_n in idx_list:
                s_n = F.pairwise_distance(h_a, h_a[h_n])#h_a[h_n] --> h-(负嵌入)
                s_n_list.append(s_n)#s_n-->d(h,h-)
            margin_label = -1 * torch.ones_like(s_p)

            loss_mar = 0
            loss_mar_1 = 0
            mask_margin_N = 0
            for s_n in s_n_list:
                loss_mar += (margin_loss(s_p, s_n, margin_label)).mean()
                loss_mar_1 += (margin_loss(s_p_1, s_n, margin_label)).mean()
                mask_margin_N += torch.max((s_n - s_p.detach() - my_margin_2), lbl_z).sum()
            mask_margin_N = mask_margin_N / num_neg

            # loss =  loss_mar_1 * args.w_loss2 + mask_margin_N * args.w_loss3#没有1

            # loss = loss_mar * args.w_loss1 + mask_margin_N * args.w_loss3  # 没有2
            #
            loss = loss_mar * args.w_loss1 + loss_mar_1 * args.w_loss2 + mask_margin_N * args.w_loss3

            loss.backward()
            optimiser.step()
            string_1 = " loss_1: {:.3f}||loss_2: {:.3f}||loss_3: {:.3f}||".format(loss_mar.item(), loss_mar_1.item(),
                                                                                  mask_margin_N.item())
            if args.save_model:
                torch.save(model.state_dict(), args.checkpoint_dir + '/' + args.dataset_name + '_weights.pth')
            if epoch % args.epochs == 0 and epoch != 0:
                model.eval()
                h_a, h_p = model.embed(feature_X, A_I_nomal)#不经过dropout
                # embs = h_a
                embs = h_p
                embs = embs / embs.norm(dim=1)[:, None]



                kmeans = KMeans(n_clusters=nb_classes).fit(embs[ylablesx].cpu().detach().numpy())
                predict_labels = kmeans.predict(embs[ylablesx].cpu().detach().numpy())

                indx = np.where(gt_reshape != 0)
                labels = gt_reshape[indx]

                pixel_y = spixel_to_pixel_labels(predict_labels, Q)
                prediction2 = pixel_y[indx]

                acc, kappa, nmi, ari, pur, ca , y_best= cluster_accuracy(labels, prediction2, return_aligned=False)

                print('k-means --- ACC: %5.4f, NMI: %5.4f' % (acc, nmi))
                for i in range(nb_classes):
                    print('class_%d:' % (i + 1), end='')
                    print('(%.2f)' % (((np.where(gt_reshape == i + 1)[0]).shape[0]) / ((indx[0]).shape[0]) * 100), end=' ')
                    print(ca[i])

                f = open('./results/' + dataset_name + '_results.txt', 'a+')


                str_results = '\n\n************************************************' \
                              + '\nacc={:.4f}'.format(np.mean(acc)) \
                              + '\nkappa={:.4f}'.format(np.mean(kappa)) \
                              + '\nnmi={:.4f}'.format(np.mean(nmi)) \
                              + '\nari={:.4f}'.format(np.mean(ari)) \
                              + '\npur={:.4f}'.format(np.mean(pur)) \
                              + '\nca=' + str(np.around(ca, 4)) \


                f.write(str_results)
                f.close()

                cprint("Done")

        palette = {0: (255, 255, 255)}
        if dataset_name == 'Indian':

            for k, color in enumerate(sns.color_palette("hls", num_classes + 1)):
                palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype='uint8'))

        elif dataset_name == 'PaviaU':

            for k, color in enumerate(sns.color_palette("Paired", num_classes + 1)):
                palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype='uint8'))

        elif dataset_name == 'Salinas':

            for k, color in enumerate(sns.color_palette("Paired", 12)):
                palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype='uint8'))

            for k, color in enumerate(sns.color_palette("hls", 4)):
                palette[13 + k] = tuple(np.asarray(255 * np.array(color), dtype='uint8'))

        elif dataset_name == 'Trento':

            for k, color in enumerate(sns.color_palette("hls", num_classes + 1)):
                palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype='uint8'))


        prediction=np.array(gt)
        # prediction[total_indices[:,0],total_indices[:,1]]= labels.astype(np.uint8)+1 #测试集的位置 测试集的输出

        prediction[np.where(gt != 0)] = y_best.astype(np.uint8)  # 测试集的位置 测试集的输出
        colored_gt = color_results(gt, palette)
        colored_pred = color_results(prediction, palette)

        outfile = os.path.join("./results", dataset_name)
        os.makedirs(outfile, exist_ok=True)
        imageio.imsave(os.path.join("./results", dataset_name +str(m) + '_gt.png'), colored_gt)  # eps or png
        imageio.imsave(os.path.join("./results", dataset_name +str(m) + '_out.png'), colored_pred)  # or png


if __name__ == '__main__':

    main_args = get_args( model_name="HSI")
    pprint_args(main_args)
    dataset = 'Salinas'  # ['Indian', 'PaviaU', 'Pavia', 'Salinas', 'KSC', 'Botswana', 'Houston', 'Trento'],
    input, num_classes, y_true, gt_reshape, gt_hsi = get_data(dataset)  # Indian Salinas
    # normalize data by band norm
    input_normalize = normalize(input)
    height, width, band = input_normalize.shape  # 145*145*200
    print("height={0},width={1},band={2}".format(height, width, band))
    input_numpy = np.array(input_normalize)

    superpixel_scale = 250
    ls = LDA_SLIC.LDA_SLIC(input_numpy,gt_hsi, num_classes - 1)
    Q, S, A, Edge_index, Edge_atter, Seg = ls.simple_superpixel(scale=superpixel_scale)
    A = torch.from_numpy(A).to(device)
    Edge_index = torch.from_numpy(Edge_index)
    Edge_atter = torch.from_numpy(Edge_atter)
    SP_size = Q.shape[1]
    ################Q是超像素的蒙版 145*145*1 范围是1-199######
    ################S是超像素的特征值 199*15 #################
    ################A是邻接矩阵  199*199 ####################
    data = data_process(S, Edge_index, Edge_atter, y_true, SP_size)



    run_S2GCL(main_args, data,gt_reshape,Q,dataset, gt_hsi)



