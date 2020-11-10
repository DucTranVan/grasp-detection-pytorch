import torch
import PIL
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from shapely.geometry import Polygon
from math import pi
import torchvision
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy

class Cornell_Grasp_dataset(Dataset):
    def __init__(self, path2data, transform, transform_params):
        self.df = pd.read_csv(path2data)  
        self.transform = transform
        self.transform_params = transform_params 

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        path = self.df.iloc[idx][0]
        label = self.df.iloc[idx][1]
        
        img = Image.open(path)
        label_list = eval(label) # in pandas label object is tring '[...]'
        label_tensor = torch.tensor(label_list)

        img, label_tensor = self.transform(img, label_tensor, self.transform_params)

        return img, label_tensor
        
def bboxes_to_grasps(bboxes):
    """Convert boxes to grasp representation, 
    which is in format tensor([x, y, theta, h, w])

    """
    x = bboxes[:,0] + (bboxes[:,4] - bboxes[:,0])/2
    y = bboxes[:,1] + (bboxes[:,5] - bboxes[:,1])/2 
    theta = torch.atan((bboxes[:,3] -bboxes[:,1]) / (bboxes[:,2] -bboxes[:,0]))
    w = torch.sqrt(torch.pow((bboxes[:,2] -bboxes[:,0]), 2) + torch.pow((bboxes[:,3] -bboxes[:,1]), 2))
    h = torch.sqrt(torch.pow((bboxes[:,6] -bboxes[:,0]), 2) + torch.pow((bboxes[:,7] -bboxes[:,1]), 2))
    grasps = torch.stack((x, y, theta, h, w), 1)
    return grasps

def grasps_to_bboxes(grasps):
    """ convert grasp representation to box
    
    """
    x = grasps[:,0]
    y = grasps[:,1]
    theta = grasps[:,2]
    h = grasps[:,3]
    w = grasps[:,4]
    x1 = x -w/2*torch.cos(theta) +h/2*torch.sin(theta)
    y1 = y -w/2*torch.sin(theta) -h/2*torch.cos(theta)
    x2 = x +w/2*torch.cos(theta) +h/2*torch.sin(theta)
    y2 = y +w/2*torch.sin(theta) -h/2*torch.cos(theta)
    x3 = x +w/2*torch.cos(theta) -h/2*torch.sin(theta)
    y3 = y +w/2*torch.sin(theta) +h/2*torch.cos(theta)
    x4 = x -w/2*torch.cos(theta) -h/2*torch.sin(theta)
    y4 = y -w/2*torch.sin(theta) +h/2*torch.cos(theta)
    bboxes = torch.stack((x1, y1, x2, y2, x3, y3, x4, y4), 1)
    return bboxes

def box_iou(bbox_value, bbox_target):
    """Helper function to calculate IoU beetween two rotation rectangles
    
    """
    p1 = Polygon(bbox_value.view(-1,2).tolist())
    p2 = Polygon(bbox_target.view(-1,2).tolist())
    iou = p1.intersection(p2).area / (p1.area +p2.area -p1.intersection(p2).area)
    return iou


class pipeline():
    def __init__(self, model, params, device):
        self.params = params
        self.model = model
        self.device = device

    def metrics_batch(self, output, target):
        pre_bboxes = grasps_to_bboxes(output)
        count = 0
        if target.shape[1] == 8:
            target_grasps = bboxes_to_grasps(target)  
            for i in range(len(output)):  
                iou = box_iou(pre_bboxes[i], target[i])
                pre_theta = output[i][2]
                target_theta = target_grasps[i][2]
                angle_diff = torch.abs(pre_theta - target_theta)*180/pi
                if angle_diff < 30 and iou > 0.25:
                    count = count + 1
        else:
            good = [0 for i in range(len(output))]
            all_grasps = bboxes_to_grasps(target[:, 1:])
            for i in range(len(target)):
                index = target[i][0].int()
                if good[index] == 1:
                    continue
                iou = box_iou(pre_bboxes[index], target[i][1:])
                pre_theta = output[index][2]
                target_theta = all_grasps[i][2]
                angle_diff = torch.abs(pre_theta - target_theta)*180/pi
                if angle_diff < 30 and iou > 0.25:
                    good[index] = 1
            for flag in good:
                if flag == 1:
                    count = count + 1

        return count

    def train_loss_batch(self, output, targets):
        params = self.params
        params_loss = params["params_loss"]
        mse_loss = params_loss["mse_loss"]
        gama = params_loss["gama"]
        opt = params["optimizer"]

        grasps = bboxes_to_grasps(targets)
        loss_x =mse_loss(output[:,0], grasps[:,0])
        loss_y =mse_loss(output[:,1], grasps[:,1])
        loss_h =mse_loss(output[:,3], grasps[:,3])
        loss_w =mse_loss(output[:,4], grasps[:,4])
        loss_theta = mse_loss(output[:,2], grasps[:,2])
        loss = loss_x + loss_y + loss_h + loss_w + gama*loss_theta

        # get performance metric
        metric_b = self.metrics_batch(output,targets)
        
        # update parameters
        opt.zero_grad()
        loss.backward()
        opt.step()

        return loss.item(), metric_b

    def val_loss_batch(self, output,targets):
        params = self.params
        params_loss = params["params_loss"]
        loss = 0.0
        gama = params_loss["gama"]

        for index, grasp in enumerate(output):
            target = [g[1:] for g in targets if g[0].int() == index]
            target = torch.stack(target)
            target_grasps = bboxes_to_grasps(target)
            loss_x = torch.pow((target_grasps[:,0] - grasp[0]), 2).mean()
            loss_y = torch.pow((target_grasps[:,1] - grasp[1]), 2).mean()
            loss_h = torch.pow((target_grasps[:,3] - grasp[3]), 2).mean()
            loss_w = torch.pow((target_grasps[:,4] - grasp[4]), 2).mean()
            loss_theta = torch.pow((target_grasps[:,2] - grasp[2]), 2).mean()
            loss = loss + loss_x + loss_y + loss_h + loss_w + gama*loss_theta

        # get performance metric
        metric_b = self.metrics_batch(output,targets)


        return loss, metric_b
    
    def _loss_epoch(self, training=True):
        params = self.params
        sanity_check = params["sanity_check"]
        train_dl = params["train_dl"]
        val_dl = params["val_dl"]
        model = self.model
        running_loss=0.0
        running_metric=0.0

        if training:
            dataset_dl = train_dl
        else:
            dataset_dl = val_dl

        len_data=len(dataset_dl.dataset)

        for xb, yb in dataset_dl:
            yb=yb.to(self.device)

            # get model output
            output = model(xb.to(self.device))

            # get loss per batch
            if training:
                loss_b, metric_b = self.train_loss_batch(output, yb)
            else:
                loss_b, metric_b = self.val_loss_batch(output, yb)

            # update running loss
            running_loss += loss_b

            # update running metric
            running_metric += metric_b

            if sanity_check:
                break

        # average loss value
        loss = running_loss/float(len_data)

        # average metric value
        metric = running_metric/float(len_data)

        return loss, metric

    def _get_lr(self, opt): 
        for param_group in opt.param_groups:
            return param_group['lr']

    def train_val(self):
        params = self.params
        model = self.model
        num_epochs = params["num_epochs"]
        lr_scheduler = params["lr_scheduler"]
        path2weights = params["path2weights"]
        opt = params["optimizer"]
        
        # history of loss values in each epoch
        loss_history={
            "train": [],
            "val": [],
        }
    
        # histroy of metric values in each epoch
        metric_history={
            "train": [],
            "val": [],
        }       
    
    
        # a deep copy of weights for the best performing model
        best_model_wts = copy.deepcopy(model.state_dict())
    
        # initialize best loss to a large value
        best_loss=float('inf')    
    
        for epoch in range(num_epochs):
            # get current learning rate
            current_lr = self._get_lr(opt)
            print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs - 1, current_lr))   

            # train the model
            model.train()
            train_loss, train_metric = self._loss_epoch(training=True)

            # collect loss and metric for training dataset
            loss_history["train"].append(train_loss)
            metric_history["train"].append(train_metric)
        
            # evaluate the model
            model.eval()
            with torch.no_grad():
                val_loss, val_metric = self._loss_epoch(training=False)
       
            # collect loss and metric for validation dataset
            loss_history["val"].append(val_loss)
            metric_history["val"].append(val_metric)   
        
        
            # store best model
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            
                # store weights into a local file
                torch.save(model.state_dict(), path2weights)
                print("Copied best model weights!")
            
            # learning rate schedule
            lr_scheduler.step(val_loss)
            if current_lr != self._get_lr(opt):
                print("Loading best model weights!")
                model.load_state_dict(best_model_wts) 
            

            print("train loss: %.6f, accuracy: %.2f" %(train_loss,100*train_metric))
            print("val loss: %.6f, accuracy: %.2f" %(val_loss,100*val_metric))
            print("-"*10) 
        

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model, loss_history, metric_history


