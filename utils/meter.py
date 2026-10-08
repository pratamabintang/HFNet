import json
import os
import torch
import numpy as np


                                                              
                      
                                     
                                
                                                                        
                                                                                   


def confusion_matrix(x, y, n, ignore_label=None, mask=None):
           
                  
    x = np.asarray(x)
    y = np.asarray(y)

            
    x = x.astype(np.int64)
    y = y.astype(np.int64)

          
    if mask is None:
        mask = np.ones_like(x, dtype=bool)
    else:
        mask = mask.astype(bool)

               
    valid_pixels = x >= 0
    if ignore_label is not None:
        valid_pixels = valid_pixels & (x != ignore_label)

    valid_pixels = valid_pixels & (y < n) & mask

            
    indices = n * x[valid_pixels].astype(int) + y[valid_pixels]
    return np.bincount(indices, minlength=n**2).reshape(n, n)

def getScores(conf_matrix, fg_only=True):
    """
    Computes segmentation performance metrics.
    When fg_only=True (default), IoU is calculated strictly for the foreground landslide class (class 1).
    """
    if conf_matrix.sum() == 0:
        return 0, 0, 0
    with np.errstate(divide="ignore", invalid="ignore"):
        overall = np.diag(conf_matrix).sum() / float(conf_matrix.sum())
        perclass = np.diag(conf_matrix) / conf_matrix.sum(1).astype(np.float64)
        IU = np.diag(conf_matrix) / (conf_matrix.sum(1) + conf_matrix.sum(0) - np.diag(conf_matrix)).astype(np.float64)

        fg_iou = float(IU[1] * 100.0) if len(IU) > 1 else float(IU[0] * 100.0)
        mean_iou = float(np.nanmean(IU) * 100.0)
        iou_out = fg_iou if fg_only else mean_iou

    return overall * 100.0, np.nanmean(perclass) * 100.0, iou_out


def get_foreground_iou(conf_matrix, fg_class=1):
    """Computes IoU exclusively for the foreground landslide class."""
    if conf_matrix.shape[0] <= fg_class:
        return 0.0
    tp = conf_matrix[fg_class, fg_class]
    union = np.sum(conf_matrix[fg_class, :]) + np.sum(conf_matrix[:, fg_class]) - tp
    if union > 0:
        return float(tp / union * 100.0)
    return 0.0


def compute_params(model):
                                      
    n_total_params = 0
    for name, m in model.named_parameters():
        n_elem = m.numel()
        n_total_params += n_elem
    return n_total_params


                                                                                         
class AverageMeter(object):
                                                           

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Saver:
                                             

    def __init__(self, args, ckpt_dir, best_val=0, condition=lambda x, y: x > y):
                   
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        with open("{}/args.json".format(ckpt_dir), "w") as f:
            json.dump(
                {k: v for k, v in args.items() if isinstance(v, (int, float, str))},
                f,
                sort_keys=True,
                indent=4,
                ensure_ascii=False,
            )
        self.ckpt_dir = ckpt_dir
        self.best_val = best_val
        self.condition = condition
        self._counter = 0

    def _do_save(self, new_val):
                                        
        return self.condition(new_val, self.best_val)

    def save(self, new_val, dict_to_save):
                                 
        self._counter += 1
        if self._do_save(new_val):
                                                                                                    
            self.best_val = new_val
            dict_to_save["best_val"] = new_val
            torch.save(dict_to_save, "{}/model-best.pth.tar".format(self.ckpt_dir))
        else:
            dict_to_save["best_val"] = new_val
            torch.save(dict_to_save, "{}/checkpoint.pth.tar".format(self.ckpt_dir))
