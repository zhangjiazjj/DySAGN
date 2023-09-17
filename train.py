import argparse
import time
import dgl
import torch
import torch.nn.functional as F
from dataset import EllipticDataset
from model import EvolveGCNH, EvolveGATO
from utils import Measure
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def train(args, device):
    elliptic_dataset = EllipticDataset(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        self_loop=True,
        reverse_edge=True,
    )

    g, node_mask_by_time = elliptic_dataset.process()
    num_classes = elliptic_dataset.num_classes

    cached_subgraph = []
    cached_labeled_node_mask = []
    for i in range(len(node_mask_by_time)):
        # we add self loop edge when we construct full graph, not here
        node_subgraph = dgl.node_subgraph(graph=g, nodes=node_mask_by_time[i])
        cached_subgraph.append(node_subgraph.to(device))
        valid_node_mask = node_subgraph.ndata["label"] >= 0
        cached_labeled_node_mask.append(valid_node_mask)

    if args.model == "EvolveGAT-O":
        model = EvolveGATO(
            in_feats=int(g.ndata["feat"].shape[1]),
            n_hidden=args.n_hidden,
            num_layers=args.n_layers,
        )
    elif args.model == "EvolveGCN-H":
        model = EvolveGCNH(
            in_feats=int(g.ndata["feat"].shape[1]), num_layers=args.n_layers
        )
    else:
        return NotImplementedError("Unsupported model {}".format(args.model))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # split train, valid, test(0-30,31-35,36-48)
    # train/valid/test split follow the paper.

    # 0-33，34-39，34-48
    train_max_index = 30
    valid_max_index = 35
    test_max_index = 48
    time_window_size = args.n_hist_steps
    loss_class_weight = [float(w) for w in args.loss_class_weight.split(",")]
    loss_class_weight = torch.Tensor(loss_class_weight).to(device)

    train_measure = Measure(
        num_classes=num_classes, target_class=args.eval_class_id
    )
    valid_measure = Measure(
        num_classes=num_classes, target_class=args.eval_class_id
    )
    test_measure = Measure(
        num_classes=num_classes, target_class=args.eval_class_id
    )

    test_res_f1 = 0
    for epoch in range(args.num_epochs):
        model.train()
        for i in range(time_window_size, train_max_index + 1):
            g_list = cached_subgraph[i - time_window_size: i + 1]
            predictions = model(g_list)
            # get predictions which has label
            predictions = predictions[cached_labeled_node_mask[i]]
            labels = (
                cached_subgraph[i]
                .ndata["label"][cached_labeled_node_mask[i]]
                .long()
            )
            loss = F.cross_entropy(
                predictions, labels, weight=loss_class_weight
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # eval
        model.eval()

        all_feature = []
        all_label = []

        for i in range(time_window_size, train_max_index + 1):
            g_list = cached_subgraph[i - time_window_size: i + 1]
            fet = g_list[-1].ndata["feat"][cached_labeled_node_mask[i]]
            predictions = model(g_list)
            # get node predictions which has label
            predictions = predictions[cached_labeled_node_mask[i]]
            labels = (
                cached_subgraph[i]
                .ndata["label"][cached_labeled_node_mask[i]]
                .long()
            )

            predictions = torch.cat((fet, predictions), dim=1)
            predictions = predictions.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            all_feature.extend(predictions)
            all_label.extend(labels)

        # get each epoch measure during eval.

        all_feature = np.array(all_feature)
        all_label = np.array(all_label)

        rf = RandomForestClassifier(n_estimators=100)

        print("train rf ...")
        rf.fit(all_feature, all_label)

        all_val_feature = []
        all_val_label = []
        for i in range(train_max_index + 1, valid_max_index + 1):
            g_list = cached_subgraph[i - time_window_size: i + 1]
            fet = g_list[-1].ndata["feat"][cached_labeled_node_mask[i]]
            predictions = model(g_list)
            # get node predictions which has label
            predictions = predictions[cached_labeled_node_mask[i]]
            labels = (
                cached_subgraph[i]
                .ndata["label"][cached_labeled_node_mask[i]]
                .long()
            )
            predictions = torch.cat((fet, predictions), dim=1)
            predictions = predictions.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            all_val_feature.extend(predictions)
            all_val_label.extend(labels)

        all_val_feature = np.array(all_val_feature)
        all_val_label = np.array(all_val_label)

        predictions = rf.predict(all_val_feature)

        # 计算指标
        precision = precision_score(all_val_label, predictions, pos_label=1)
        recall = recall_score(all_val_label, predictions, pos_label=1)
        f1 = f1_score(all_val_label, predictions, pos_label=1)
        Mf1 = f1_score(all_val_label, predictions, average='micro')
        print(
            "Eval Epoch {} | class {} | precision:{:.4f} | recall: {:.4f} | f1: {:.4f} | f1: {:.4f}".format(
                epoch, 1, precision, recall, f1,Mf1
            )
        )

        if f1 >= test_res_f1:
            print(
                "###################Epoch {} Test###################".format(
                    epoch
                )
            )
            test_res_f1 = f1

            all_test_feature = []
            all_test_label = []
            for i in range(valid_max_index + 1, test_max_index + 1):
                g_list = cached_subgraph[i - time_window_size: i + 1]
                fet = g_list[-1].ndata["feat"][cached_labeled_node_mask[i]]
                predictions = model(g_list)
                # get node predictions which has label
                predictions = predictions[cached_labeled_node_mask[i]]
                labels = (
                    cached_subgraph[i]
                    .ndata["label"][cached_labeled_node_mask[i]]
                    .long()
                )
                predictions = torch.cat((fet, predictions), dim=1)
                predictions = predictions.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
                all_test_feature.extend(predictions)
                all_test_label.extend(labels)

                predictions_time = rf.predict(predictions)
                precision_time = precision_score(labels, predictions_time, pos_label=1)
                recall_time = recall_score(labels, predictions_time, pos_label=1)
                f1_time = f1_score(labels, predictions_time, pos_label=1)
                Mf1_time= f1_score(labels, predictions_time, average='micro')

                print(f"Test | Time {i + 1}")
                print(f"Precision: {precision_time:.4f}, Recall: {recall_time:.4f}, F1: {f1_time:.4f}, MF1: {Mf1_time:.4f}")

            all_test_feature = np.array(all_test_feature)
            all_test_label = np.array(all_test_label)

            pre_all = rf.predict(all_test_feature)
            precision_test_all = precision_score(all_test_label, pre_all, pos_label=1)
            recall_test_all = recall_score(all_test_label, pre_all, pos_label=1)
            f1_test_all = f1_score(all_test_label, pre_all, pos_label=1)
            Mf1_test_all = f1_score(all_test_label, pre_all, average='micro')

            print(
                "  Test | Epoch {} | class {} | precision:{:.4f} | recall: {:.4f} | f1: {:.4f} | f1: {:.4f}".format(
                    epoch, 1, precision_test_all, recall_test_all, f1_test_all,Mf1_test_all
                )
            )

    print(
        "Best test f1 is {}".format(
            test_res_f1
        )
    )

if __name__ == "__main__":
    argparser = argparse.ArgumentParser("EvolveGCN")
    argparser.add_argument(
        "--model",
        type=str,
        default="EvolveGAT-O",
        help="We can choose EvolveGCN-O or EvolveGCN-H,"
        "but the EvolveGCN-H performance on Elliptic dataset is not good.",
    )
    argparser.add_argument(
        "--raw-dir",
        type=str,
        default="../data/epllic",
        help="Dir after unzip downloaded dataset, which contains 3 csv files.",
    )
    argparser.add_argument(
        "--processed-dir",
        type=str,
        default="../data/process/",
        help="Dir to store processed raw data.",
    )
    argparser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID. Use -1 for CPU training.",
    )
    argparser.add_argument("--num-epochs", type=int, default=1000)
    argparser.add_argument("--n-hidden", type=int, default=180)
    argparser.add_argument("--n-layers", type=int, default=2)
    argparser.add_argument(
        "--n-hist-steps",
        type=int,
        default=7,
        help="If it is set to 5, it means in the first batch,"
        "we use historical data of 0-4 to predict the data of time 5.",
    )
    argparser.add_argument("--lr", type=float, default=0.001)
    argparser.add_argument(
        "--loss-class-weight",
        type=str,
        default="0.35,0.65",
        help="Weight for loss function. Follow the official code,"
        "we need to change it to 0.25, 0.75 when use EvolveGCN-H",
    )
    argparser.add_argument(
        "--eval-class-id",
        type=int,
        default=1,
        help="Class type to eval. On Elliptic, type 1(illicit) is the main interest.",
    )
    argparser.add_argument(
        "--patience", type=int, default=100, help="Patience for early stopping."
    )

    args = argparser.parse_args()

    if args.gpu >= 0:
        device = torch.device("cuda:%d" % args.gpu)
    else:
        device = torch.device("cpu")

    start_time = time.perf_counter()
    train(args, device)
    print("train time is: {}".format(time.perf_counter() - start_time))