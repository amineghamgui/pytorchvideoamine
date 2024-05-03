import torch
import json
import urllib
from pytorchvideo.data.encoded_video import EncodedVideo

from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)
import wandb
import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image

import torch
from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score




#############################dataset & data loader###########################


class X3D_Dataset(Dataset):
    def __init__(self,
                 
                 is_train=False,
                 img_size=224,
                 transform=None):
        self._downsample = 4
        self.num_classes = 2            
        if is_train:
            self.pathhhhh = os.path.join("/kaggle/input/train-csv/train.csv")
            
        else:
            self.pathhhhh = os.path.join("/kaggle/input/val-csv/val.csv")

        self.transform = transform
        self.is_train = is_train
        self.img_size = img_size
        
        self._load_data()

    
    
  

    import cv2
    def parse_csv_to_dict(self):
        result_dict = {}

        with open(self.pathhhhh, "r") as f:
#             f.readline()  # Ignorer la première ligne (en-tête)

            for line in f:
                row = line.strip().split(",")
                if (int(row[6])==0) or (int(row[6])==1) or (int(row[6])==2):
                    key = row[0]
                    key1 = row[1]
                    if key not in result_dict:
                        result_dict[key] = {}

                    path = "/kaggle/input/data-faux-train-yowo/data_faux/" + key + "/" + key1 + ".mp4"
                    result_dict[key][key1] = EncodedVideo.from_path(path)

        return result_dict

    
    def get_labels_to_seq(self):
        result_dict = {}

        with open(self.pathhhhh, "r") as f:
#             f.readline()
            for line in f:

                row = line.strip().split(",")
                if (int(row[6])==0) or (int(row[6])==1) or (int(row[6])==2):
                    key = row[0]
                    key1 = row[1]
                    if key not in result_dict:
                        result_dict[key] = {}  
                    if key1 not in result_dict[key]:
                        result_dict[key][key1] = [] 
                    if (int(row[6])==0 ):
                        result_dict[key][key1].append(0)
                    elif (int(row[6])==1) or (int(row[6])==2):
                        
                        result_dict[key][key1].append(1)
        return result_dict
    
    @staticmethod  
    def class_to_list(l,numclasses):
        l_entiers = [int(element) for element in l]
        liste0=[0 for i in range(numclasses)]
        for i in l_entiers:
            liste0[i]=1
        return liste0
    
    def _load_data(self):
        video_factory=self.parse_csv_to_dict()
        annotation_factory=self.get_labels_to_seq()
        self.l_clip=[]
        self.l_labels=[]
        
        
        for keys in annotation_factory:
            for keys1 in annotation_factory[keys]:
                
                self.l_clip.append(video_factory[keys][keys1])
                self.l_labels.append(annotation_factory[keys][keys1])
        print(self.l_labels)
         
    def __len__(self):
        return len(self.l_labels)



    def __getitem__(self, idx):
        frame_idx, video_clip, target = self.pull_item(idx)

        return frame_idx, video_clip, target
    
    def pull_item(self, idx):
        seq = self.l_clip[idx]
        keyframe_info = self.l_clip[idx] 

        labels = self.l_labels[idx]
#         annotations= self.class_to_list(labels,self.num_classes)
        annotations=self.class_to_list(labels,self.num_classes)
        clip=self.l_clip[idx]
        
        start_sec = 0
        end_sec = start_sec + clip_duration


        # Load the desired clip
        video_data = clip.get_clip(start_sec=start_sec, end_sec=end_sec)
        
        l_clip = self.transform(video_data)
        l_clip=l_clip["video"]
        # Reformater target
        target = {
             #'labels':  torch.tensor([annotations]),  
            'labels': int(labels[0]),
             'sec': idx,

        }

        return keyframe_info, l_clip, target  

#########################################Focal loss###########################################"


class SigmoidFocalLoss(object):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    """

    def __init__(self, num_classes: int, alpha: float = 0.25, gamma: float = 2, reduction: str = "none"):
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def __call__(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (Tensor): A float tensor of shape (N, C), where N is the number of samples
                and C is the number of classes.
            targets (Tensor): A long tensor of shape (N,), containing the class labels
                for each sample.

        Returns:
            Loss tensor with the reduction option applied.
        """
        p = torch.sigmoid(inputs)
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")  # Cross entropy loss
        p_t = p[torch.arange(len(targets)), targets]  # Get the probabilities of the target class
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * torch.ones_like(targets, dtype=torch.float32)
            loss = alpha_t * loss

        # Check reduction option and return loss accordingly
        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{self.reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
            )
        return loss


###############################collatefunction###############


class CollateFunc(object):
    def __call__(self, batch):
        batch_frame_id = []
        batch_key_target = []
        batch_video_clips = []

        for sample in batch:
            key_frame_id = sample[0]
            video_clip = sample[1]
            key_target = sample[2]['labels']
            
            batch_frame_id.append(key_frame_id)
            batch_video_clips.append(video_clip)
            batch_key_target.append(key_target)

        # List [B, 3, T, H, W] -> [B, 3, T, H, W]
        batch_video_clips = torch.stack(batch_video_clips)
        batch_key_target=torch.tensor(batch_key_target)
#         print("batch_video_clips.shape",batch_video_clips.shape)
#         print("**************************************************",len(batch) )
        return batch_frame_id, batch_video_clips, batch_key_target

######################################data loader initialisation#########







if __name__ == '__main__':
    
    
    
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    frames_per_second = 30
    model_transform_params  = {
        "x3d_xs": {
            "side_size": 182,
            "crop_size": 182,
            "num_frames": 4,
            "sampling_rate": 12,
        },
        "x3d_s": {
            "side_size": 182,
            "crop_size": 182,
            "num_frames": 13,
            "sampling_rate": 6,
        },
        "x3d_m": {
            "side_size": 256,
            "crop_size": 256,
            "num_frames": 16,
            "sampling_rate": 5,
        }
    }

    # Get transform parameters based on model
    transform_params = model_transform_params["x3d_xs"]

    # Note that this transform is specific to the slow_R50 model.
    transform =  ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(transform_params["num_frames"]),
                Lambda(lambda x: x/255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(size=transform_params["side_size"]),
                CenterCropVideo(
                    crop_size=(transform_params["crop_size"], transform_params["crop_size"])
                )
            ]
        ),
    )

    # The duration of the input clip is also specific to the model.
    clip_duration = (transform_params["num_frames"] * transform_params["sampling_rate"])/frames_per_second
        
    
    img_size = 224
    len_clip = 25

    dataset_val = X3D_Dataset(

        is_train=False,
        img_size=img_size,
        transform=transform
        
    )


    dataset_train = X3D_Dataset(

        is_train=True,
        img_size=img_size,
        transform=transform
    )
    
    training_loader  = torch.utils.data.DataLoader(dataset_train,drop_last=False,
                                         pin_memory=True, batch_size=8,
                                         collate_fn=CollateFunc())
    validation_loader  = torch.utils.data.DataLoader(dataset_val,shuffle=False,drop_last=False,
                                         pin_memory=True, collate_fn=CollateFunc())
    
    
    #transfert learning 
    model_name = 'x3d_xs'
    model1 = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)
    layers = list(model1.blocks.children())
    _layers = layers[:-1]  # Extrait de caractéristiques
    classifier = layers[-1]  # Classificateur
    num_classes = 2
    classifier.proj = nn.Linear(in_features=classifier.proj.in_features, out_features=2, bias=True)
    
    #optimazer & Loss function
    optimizer = torch.optim.SGD(model1.parameters(), lr=0.001, momentum=0.9)
    loss_function = SigmoidFocalLoss(num_classes=2, alpha=0.25, gamma=2, reduction='mean')
    #Cpu or Gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1.to(device)
    
    
    #train one epoch
    def train_one_epoch(epoch_index, tb_writer):
        running_loss = 0.
        last_loss = 0.
        sum_accuracy=0
        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        concatenated_tensor_labels = torch.empty(0)
        concatenated_tensor_predicted= torch.empty(0)
        for i, data in enumerate(training_loader):
            # Every data instance is an input + label pair
            icx,inputs, labels = data
            
            inputs, labels = inputs.to(device), labels.to(device)
            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model1(inputs)
            post_act = torch.nn.Softmax(dim=1)
            preds = post_act(outputs)

            # Compute the loss and its gradients
            loss = loss_function(preds, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            
            #calcul de accuracy pour chaque batch et sommation 
            _, predicted = torch.max(outputs, 1)
            #print("train              labels     : \n",labels.cpu(),"predicted  ", predicted.cpu()) 
            accuracy=accuracy_score(labels.cpu(), predicted.cpu())
            sum_accuracy += accuracy
            
            
            concatenated_tensor_labels = torch.cat((concatenated_tensor_labels, labels.cpu()), dim=0)
            concatenated_tensor_predicted = torch.cat((concatenated_tensor_predicted, predicted.cpu()), dim=0)
            if i % 10 == 5:
                last_loss = running_loss / 10 # loss per batch

                print('  batch {} loss: {} '.format(i + 1, last_loss))
                tb_x = epoch_index * len(training_loader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.
            
        accuracy_avg=sum_accuracy/len(training_loader)
        precision=precision_score(concatenated_tensor_labels, concatenated_tensor_predicted, average='micro')
        return last_loss,accuracy_avg,precision
   
############################################affichge des courbe de train avec WandB
    
    wandb.login()
    run = wandb.init(
        # Set the project where this run will be logged
        project="X3d_project",
        # Track hyperparameters and run metadata
        config={
            
            "epochs": 5,
        },
    )
################################################################

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0

    EPOCHS = 5

    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model1.train(True)
        
        resultat_train_one_epoch= train_one_epoch(epoch_number, writer)
        avg_loss =resultat_train_one_epoch[0]
        avg_acc=resultat_train_one_epoch[1]
        precision_train=resultat_train_one_epoch[2]

        
        running_vloss = 0.0
        sum_vacc = 0.0
        concatenated_tensor_vlabels = torch.empty(0)
        concatenated_tensor_predicted= torch.empty(0)
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model1.eval()
        
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                frameidx,vinputs, vlabels = vdata
                vinputs, vlabels = vinputs.to(device), vlabels.to(device)
                voutputs = model1(vinputs)
                post_act = torch.nn.Softmax(dim=1)
                preds = post_act(voutputs)
    #             print(voutputs)
    #             print(vlabels[0]["labels"])
                vloss = loss_function(preds, vlabels)
                running_vloss += vloss
                #accuracy validation
                _, predicted = torch.max(voutputs, 1)
                #print("validation           vlabels        : \n",vlabels.cpu(),'predicted : ', predicted.cpu())
                accuracy=accuracy_score(vlabels.cpu(), predicted.cpu())
                sum_vacc=accuracy+sum_vacc
                ############################
                
                concatenated_tensor_vlabels = torch.cat((concatenated_tensor_vlabels, vlabels.cpu()), dim=0)
                #print("concatenated_tensor_vlabels\n",concatenated_tensor_vlabels)
                concatenated_tensor_predicted = torch.cat((concatenated_tensor_predicted, predicted.cpu()), dim=0)
                #print("concatenated_tensor_predicted\n",concatenated_tensor_predicted)
        precision_val=precision_score(concatenated_tensor_vlabels, concatenated_tensor_predicted, average='micro')
            
        print(classification_report(concatenated_tensor_vlabels, concatenated_tensor_predicted))
        
        accuracy_val_avg=sum_vacc/len(validation_loader)

        avg_vloss = running_vloss / (i + 1)
        
        
        
        
        print('LOSS train {} valid {} \n'.format(avg_loss, avg_vloss))
        print('acc train {} valid {} '.format(avg_acc, accuracy_val_avg))
        print('precision train {} valid {} '.format(precision_train, precision_val))
        wandb.log({"accuracy_train": avg_acc, "accuracy_val": accuracy_val_avg,"loss_train":avg_loss,"loss_val":avg_vloss,"precision_train":precision_train,"precision_val":precision_val})
        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            poid_path = 'poid_{}_{}'.format("amine", epoch_number)
            torch.save(model1.state_dict(), poid_path)#***************************************************************modifier path***********************************************************

        epoch_number += 1
