import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import f1_score
import importlib
import pdb
import pickle as pkl
import datetime
import os

def source_import(file_path):
    """This function imports python module directly from source code using importlib"""
    spec = importlib.util.spec_from_file_location('', file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def batch_show(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.figure(figsize=(20, 20))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)


def print_write(print_str, log_file):
    print(*print_str)
    with open(log_file, 'a') as f:
        print(*print_str, file=f)

#
def print_write_config(config, log_file):
    with open(log_file, 'a') as f:
        f.write(config)


def init_weights(model, weights_path, caffe=False, classifier=False):
    """Initialize weights"""
    print('Pretrained %s weights path: %s' % ('classifier' if classifier else 'feature model',
                                              weights_path))
    weights = torch.load(weights_path)
    if not classifier:
        if caffe:
            weights = {k: weights[k] if k in weights else model.state_dict()[k]
                       for k in model.state_dict()}
        else:
            weights = weights['state_dict_best']['feat_model']
            weights = {k: weights['module.' + k] if 'module.' + k in weights else model.state_dict()[k]
                       for k in model.state_dict()}
    else:
        weights = weights['state_dict_best']['classifier']
        weights = {k: weights['module.fc.' + k] if 'module.fc.' + k in weights else model.state_dict()[k]
                   for k in model.state_dict()}
    model.load_state_dict(weights)
    return model


def shot_acc(preds, labels, train_data, many_shot_thr=100, low_shot_thr=20):
    training_labels = np.array(train_data.dataset.labels).astype(int)

    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    train_class_count = []
    test_class_count = []
    class_correct = []
    for l in np.unique(labels):
        train_class_count.append(len(training_labels[training_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        class_correct.append((preds[labels == l] == labels[labels == l]).sum())

    many_shot = []
    median_shot = []
    low_shot = []
    for i in range(len(train_class_count)):
        if train_class_count[i] >= many_shot_thr:
            many_shot.append((class_correct[i] / test_class_count[i]))
        elif train_class_count[i] <= low_shot_thr:
            low_shot.append((class_correct[i] / test_class_count[i]))
        else:
            median_shot.append((class_correct[i] / test_class_count[i]))
    return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)


def get_shot_list_with_dataset(data, path='ImageNet_shots.pkl', many_shot_thr=100, low_shot_thr=20):
    labels = np.array(data.dataset.labels).astype(int)

    class_count = []
    for l in np.unique(labels):
        class_count.append(len(labels[labels == l]))

    many_shot = []
    median_shot = []
    low_shot = []
    for i in range(len(class_count)):
        if class_count[i] >= many_shot_thr:
            many_shot.append(i)
        elif class_count[i] <= low_shot_thr:
            low_shot.append(i)
        else:
            median_shot.append(i)
    dict = {"many_shot": many_shot, "median_shot":median_shot, "low_shot": low_shot}
    with open(path, 'wb') as f:
        pkl.dump(dict, f, protocol=pkl.HIGHEST_PROTOCOL)
    return dict


def get_shot_list(path='ImageNet_shots.pkl'):
    with open(path, 'rb') as f:
        shot_list = pkl.load(f)
        # shot_list['many_median_shot'] = shot_list['many_shot'] + shot_list['median_shot']
        return shot_list


def get_classwise_shot_list_with_dataset(data, path='ImageNet_classwise_shots.npy'):
    labels = np.array(data.dataset.labels).astype(int)
    class_count = []
    for l in np.unique(labels):
        class_count.append(len(labels[labels == l]))
    np.save(path, class_count)

def get_classwise_shot_list(path='ImageNet_classwise_shots.npy'):
    return np.load(path)


def argsort_idx(shot_list):
    sorted_idx = np.argsort(shot_list)
    sorted_inverse_idx = np.zeros(sorted_idx.shape)
    for i in range(len(sorted_idx)):
        sorted_inverse_idx[sorted_idx[i]] = i
    sorted_inverse_idx = sorted_inverse_idx.astype(np.int32)
    return sorted_idx, sorted_inverse_idx


def smooth_classwise_shot_list(shot_list, num_bins):
    num_class = len(shot_list)
    shot_list_sorted = np.sort(shot_list) 
    # from small to large
    sorted_idx, sorted_inverse_idx = argsort_idx(shot_list)
    shot_list_smoothed_sorted = np.zeros(shot_list.shape)
    shot_list_smoothed = np.zeros(shot_list.shape)
    num_class_per_bin = int(np.ceil(num_class / num_bins))
    for i in range(num_bins):
        start = i * num_class_per_bin
        end = min(num_class, (i + 1) * num_class_per_bin)
        shot_list_smoothed_sorted[start:end] = int(np.mean(shot_list_sorted[start:end]))
    shot_list_smoothed = shot_list_smoothed_sorted[sorted_inverse_idx]
    shot_list_smoothed = shot_list_smoothed.astype(np.int32)
    return shot_list_smoothed



def map_classid_and_label(shot_list):
    classid2label = {}
    label2classid = {}
    for label, classid in enumerate(shot_list):
        classid2label[classid] = label
        label2classid[label] = classid

    return classid2label, label2classid



def F_measure(preds, labels, openset=False, theta=None):
    if openset:
        # f1 score for openset evaluation
        true_pos = 0.
        false_pos = 0.
        false_neg = 0.

        for i in range(len(labels)):
            true_pos += 1 if preds[i] == labels[i] and labels[i] != -1 else 0
            false_pos += 1 if preds[i] != labels[i] and labels[i] != -1 and preds[i] != -1 else 0
            false_neg += 1 if preds[i] != labels[i] and labels[i] == -1 else 0

        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        return 2 * ((precision * recall) / (precision + recall + 1e-12))
    else:
        # Regular f1 score
        return f1_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro')


def mic_acc_cal(preds, labels):
    acc_mic_top1 = (preds == labels).sum().item() / len(labels)
    return acc_mic_top1


def class_count(data):
    labels = np.array(data.dataset.labels)
    class_data_num = []
    for l in np.unique(labels):
        class_data_num.append(len(labels[labels == l]))
    return class_data_num

def class_count_with_dataset(dataset='ImageNet_LT'):
    class_data_num = np.load('class_count_' + dataset + '.npy')
    return class_data_num


def map_labels_to_sorted_labels(class_shots):
    sorted_idx = np.argsort(class_shots)[::-1]
    org_label_to_sorted_label = {}
    sorted_label_to_org_label = {}
    for i in range(len(sorted_idx)):
        org_label_to_sorted_label[sorted_idx[i]] = i
        sorted_label_to_org_label[i] = sorted_idx[i]
    return org_label_to_sorted_label, sorted_label_to_org_label, sorted_idx

# def dataset_dist (in_loader):

#     """Example, dataset_dist(data['train'][0])"""

#     label_list = np.array([x[1] for x in in_loader.dataset.samples])
#     total_num = len(data_list)

#     distribution = []
#     for l in np.unique(label_list):
#         distribution.append((l, len(label_list[label_list == l])/total_num))

#     return distribution



def get_time():
    return datetime.datetime.now().strftime('%b%d_%H-%M')


def get_logfile_name(path):
    file_name = get_time() + '_log.txt'
    return os.path.join(path, file_name)
