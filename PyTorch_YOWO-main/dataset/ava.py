import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image

try:
    import ava_helper
except:
    from . import ava_helper


# Dataset for AVA
class AVA_Dataset(Dataset):
    def __init__(self,
                 cfg,
                 is_train=False,
                 img_size=224,
                 transform=None,
                 len_clip=16,
                 sampling_rate=1):
        self._downsample = 4
        self.num_classes = 5             
        # self.num_classes = 80
        self.data_root = cfg['data_root']
        self.frames_dir = os.path.join(cfg['data_root'], cfg['frames_dir'])
        self.frame_list = os.path.join(cfg['data_root'], cfg['frame_list'])
        self.annotation_dir = os.path.join(cfg['data_root'], cfg['annotation_dir'])
        self.labelmap_file = os.path.join(cfg['data_root'], cfg['annotation_dir'], cfg['labelmap_file'])
        if is_train:
            self.gt_box_list = os.path.join(self.annotation_dir, cfg['train_gt_box_list'])
            self.exclusion_file = os.path.join(self.annotation_dir, cfg['train_exclusion_file'])
        else:
            self.gt_box_list = os.path.join(self.annotation_dir, cfg['val_gt_box_list'])
            self.exclusion_file = os.path.join(self.annotation_dir, cfg['val_exclusion_file'])

        self.transform = transform
        self.is_train = is_train
        
        self.img_size = img_size
        self.len_clip = len_clip
        self.sampling_rate = sampling_rate
        self.seq_len = self.len_clip * self.sampling_rate
        self.pathhhhh="/kaggle/input/train-csv/train.csv"
        # load ava data
        self._load_data()

    
    
  

    import cv2
    @staticmethod
    def extract_frames(video_path):
        list_frames = []
        video_capture = cv2.VideoCapture(video_path)
        while True:
            success, frame = video_capture.read()
            if not success:
                break
            list_frames.append(Image.fromarray(frame))

        video_capture.release()
        return list_frames
    def parse_csv_to_dict(self):
        result_dict = {}

        with open(self.pathhhhh, "r") as f:
            f.readline()  # Ignorer la première ligne (en-tête)

            for line in f:
                row = line.strip().split(",")
                key = row[0]
                key1 = row[1]
                if key not in result_dict:
                    result_dict[key] = {}

                path = "/kaggle/input/data-faux-train-yowo/data_faux/" + key + "/" + key1 + ".mp4"
                result_dict[key][key1] = self.extract_frames(path)

        return result_dict

    
    def get_boxes_to_seq(self):
#         self.boxxx_list=parse_csv_to_dict(csv_file)
        #self.csv_file
        
        result_dict = {}

        with open(self.pathhhhh, "r") as f:
            f.readline()
            for line in f:

                row = line.strip().split(",")
                key = row[0]
                key1 = row[1]
                if key not in result_dict:
                    result_dict[key] = {}  
                if key1 not in result_dict[key]:
                    result_dict[key][key1] = []  
                result_dict[key][key1].append([row[2:6],row[6]])
        
        return result_dict
    #sortie de  get_boxes_to_seq entre de combiner_valeurs
    
    
    def combiner_valeurs(self):
        diction=self.get_boxes_to_seq()#diction 
        for keys in diction:
            for keys1 in diction[keys]:
                liste=diction[keys][keys1]
                result = {}
                for sous_liste in liste:
                    cle = tuple(map(float, sous_liste[0]))  # Conversion en float
                    valeur = int(sous_liste[1][0])  # Conversion en int
                    if cle in result:
                        result[cle].append(valeur)
                    else:
                        result[cle] = [valeur]

                resultat_final = [[list(k), v] for k, v in result.items()]
                diction[keys][keys1]=resultat_final
        return diction


    
    def _load_data(self):
        video_factory=self.parse_csv_to_dict()
        annotation_factory=self.combiner_valeurs()
        self.l_clip=[]
        self.l_boxes=[]
        for keys in annotation_factory:
            for keys1 in annotation_factory[keys]:
                print(keys)
                self.l_clip.append(video_factory[keys][keys1])
                print(keys1)
                self.l_boxes.append(annotation_factory[keys][keys1])
        
         
    def __len__(self):
        return len(self.l_boxes)


 




    def __getitem__(self, idx):
        # load a data
        frame_idx, video_clip, target = self.pull_item(idx)

        return frame_idx, video_clip, target
    


    def pull_item(self, idx):


            seq=self.l_clip[idx]
            keyframe_info=self.l_clip[idx][12]

            # load a video clip
            
            frame =self.l_clip[idx][0]
            ow, oh = frame.width, frame.height

            # Get boxes and labels for current clip.
            boxes = []
            labels = []
            for box_labels in self.l_boxes[idx]:
                bbox = box_labels[0]
                label = box_labels[1]
                multi_hot_label = np.zeros(1 + 5)
                multi_hot_label[..., label] = 1.0

                boxes.append(bbox)
                labels.append(multi_hot_label[..., 1:].tolist())

            boxes = np.array(boxes).reshape(-1, 4)
            # renormalize bbox
            boxes[..., [0, 2]] *= ow
            boxes[..., [1, 3]] *= oh
            labels = np.array(labels).reshape(-1, 5)

            # target: [N, 4 + C]
            target = np.concatenate([boxes, labels], axis=-1)

            # transform
            l_clip, target = self.transform(self.l_clip[idx], target)
            # List [T, 3, H, W] -> [3, T, H, W]
            l_clip = torch.stack(l_clip, dim=1)

            # reformat target
            target = {
                'boxes': target[:, :4].float(),  # [N, 4]
                'labels': target[:, 4:].long(),  # [N, C]
                'orig_size': [ow, oh],
                #,
                'video_idx': "video_idx",
                'sec': idx,

            }

            return keyframe_info, l_clip, target



if __name__ == '__main__':
    import cv2
    from transforms import Augmentation, BaseTransform

    is_train = False
    img_size = 224
    len_clip = 25
    dataset_config = {
        #'data_root': '/kaggle/input/data-ava/ava',
        'data_root':'/kaggle/input/ava-version2-oneclass/ava-20240312T085221Z-001/ava',
        #'data_root': '/kaggle/input/data-ava/ava',
        'frames_dir': 'frames/',
        'frame_list': 'frame_lists/',
        'annotation_dir': 'annotations/',
        'train_gt_box_list': 'ava_v2.2/ava_train_v2.2.csv',
        'val_gt_box_list': 'ava_v2.2/ava_val_v2.2.csv',
        'train_exclusion_file': 'ava_v2.2/ava_train_excluded_timestamps_v2.2.csv',
        'val_exclusion_file': 'ava_v2.2/ava_val_excluded_timestamps_v2.2.csv',
        'labelmap_file': 'ava_v2.2/ava_action_list_v2.2.pbtxt',
    }
    
    trans_config = {
        'pixel_mean': [0.45, 0.45, 0.45],
        'pixel_std': [0.225, 0.225, 0.225],
        'jitter': 0.2,
        'hue': 0.1,
        'saturation': 1.5,
        'exposure': 1.5
    }
    transform = Augmentation(
        img_size=img_size,
        pixel_mean=trans_config['pixel_mean'],
        pixel_std=trans_config['pixel_std'],
        jitter=trans_config['jitter'],
        saturation=trans_config['saturation'],
        exposure=trans_config['exposure']
        )
    # transform = BaseTransform(
    #     img_size=img_size,
    #     pixel_mean=trans_config['pixel_mean'],
    #     pixel_std=trans_config['pixel_std']
    #     )

    train_dataset = AVA_Dataset(
        cfg=dataset_config,
        is_train=is_train,
        img_size=img_size,
        transform=transform,
        len_clip=len_clip,
        sampling_rate=1
    )

    print("*************************************************************************************amine")
    print(type(train_dataset))
    print(len(train_dataset))