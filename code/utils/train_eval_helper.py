import torch
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.metrics import classification_report
import csv

def save_checkpoint(save_path, model, valid_loss):
    if save_path is None:
        return
    state_dict = {'model_state_dict': model.state_dict(), 'valid_loss': valid_loss}
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')

def load_checkpoint(load_path, model, args):
    if load_path is None:
        return
    state_dict = torch.load(load_path, map_location=args.device)
    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']

def load_partial_dict(load_path, model, args):
    state_dict = torch.load(load_path, map_location=args.device)
    model.load_state_dict(state_dict['model_state_dict'], strict=False)
    print(f'Model partial parameters loaded from <== {load_path}')

def draw_fig_loss(loss_list_train, loss_list_val, global_steps_list, save_dir):
    plt.cla()
    plt.plot(global_steps_list, loss_list_train, label='Train Loss')
    plt.plot(global_steps_list, loss_list_val, label='Valid Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, 'train_valid_loss.jpg'), dpi=300)

def draw_fig_acc(acc_list_train, acc_list_val, global_steps_list, save_dir):
    plt.cla()
    plt.plot(global_steps_list, acc_list_train, label='Train Acc.')
    plt.plot(global_steps_list, acc_list_val, label='Valid Acc.')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, 'train_valid_acc.jpg'), dpi=300)

def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def eval_classification_report(out_log, do_print=False):
    pred_y_list = []
    y_list = []
    for batch in out_log:
        pred_y = batch[0].detach().cpu().numpy().argmax(axis=1).tolist()
        y = batch[1].detach().cpu().numpy().tolist()
        pred_y_list.extend(pred_y)
        y_list.extend(y)
    out = classification_report(y_list, pred_y_list, labels=[1,0], target_names=['Fake','True'], digits=4, output_dict=(not do_print))
    if do_print:
        print(out)
    return out

def create_out_csv(root, name):
    path = os.path.join(root, name + '_out.csv')
    with open(path, 'w', newline='') as f:
        csv_write = csv.writer(f)
        csv_head = ['Args','Accuracy','Mac.F1','F-P','F-R','F-F1','T-P','T-R','T-F1']
        csv_write.writerow(csv_head)

def append_out_csv(root, name, out, args):
    path = os.path.join(root, name + '_out.csv')
    with open(path, 'a+', newline='') as f:
        csv_write = csv.writer(f)
        data_row = [args, "{:.4f}".format(out['accuracy']), "{:.4f}".format(out['macro avg']['f1-score']),
                    "{:.4f}".format(out['Fake']['precision']), "{:.4f}".format(out['Fake']['recall']),
                    "{:.4f}".format(out['Fake']['f1-score']), "{:.4f}".format(out['True']['precision']),
                    "{:.4f}".format(out['True']['recall']), "{:.4f}".format(out['True']['f1-score'])]
        csv_write.writerow(data_row)

class LRScheduler:
    def __init__(self, optimizer, patience=4, min_lr=1e-6, factor=0.5):
        self.optimizer = optimizer
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=patience, factor=factor, min_lr=min_lr, verbose=True)
    def __call__(self, val_acc):
        self.lr_scheduler.step(val_acc)

class EarlyStoppingAcc:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_acc = float("Inf")
        self.early_stop = False
    def __call__(self, val_acc):
        if self.best_acc == float("Inf"):
            self.best_acc = val_acc
        elif self.best_acc - val_acc < self.min_delta:
            self.best_acc = val_acc
            self.counter = 0
        elif self.best_acc - val_acc > self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True