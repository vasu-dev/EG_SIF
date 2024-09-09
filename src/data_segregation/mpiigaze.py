### IMPORT ###

import torch
torch.manual_seed(0)
from torchvision import transforms
from torch.utils.data import Dataset, ConcatDataset

import os
import cv2
import h5py
import glob
import scipy.io
import numpy as np
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings("ignore")
### END IMPORT 

### CONFIG ###
persons = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14"]
#persons = ["00"]
def save_on_disk(dump_folder_path, person, dataloader, iq="good"):
    
    imgs = []
    gazes = []
    poses = []

    for batch in tqdm(dataloader):    
        img, gaze, pose = batch[0], batch[1], batch[3]
        img = img.numpy()
        gaze = gaze.numpy() 
        pose = pose.numpy()
        imgs.extend(img)
        gazes.extend(gaze)
        poses.extend(pose)
        
        
    imgs = np.array(imgs, dtype=np.uint8)
    gazes = np.array(gazes)
    poses = np.array(poses)

    bin_name = dump_folder_path + "{}.h5".format(iq)
    f = h5py.File(bin_name,'w')
    f.create_dataset('imgs', shape = imgs.shape, data= imgs)
    f.create_dataset('gaze', shape = gazes.shape, data= gazes)
    f.create_dataset('pose', shape = poses.shape, data= poses)
    f.close()
    print ("person : {}, {} data saved to path : {}".format(person, iq, bin_name))
    

## Data Segregation Class
class MPIIGazeSegregation(Dataset): # custom data loader for MPIIGAZE normalized data
    def __init__(self, data_path, eye):
        self.eye = eye.lower()
        
        assert self.eye in ["left", "right"], "The value of eye should be either left or right"
        self.image_list = []
        self.labels_list = []
        self.pose_list = []
        
        for mat_file in glob.glob(data_path + "*/*.mat"):
            mat = scipy.io.loadmat(mat_file)
            if self.eye == 'left':
                images = mat['data'][0][0][0][0][0][1]
                gaze = mat['data'][0][0][0][0][0][0]
                pose = mat['data'][0][0][0][0][0][2]
            else:
                images = mat['data'][0][0][1][0][0][1]
                gaze = mat['data'][0][0][1][0][0][0]
                pose = mat['data'][0][0][1][0][0][2]
            
            self.image_list.extend(images)
            self.labels_list.extend(gaze)
            self.pose_list.extend(pose)
        
    def convert_pose(self, vector: np.ndarray) -> np.ndarray:
        rot = cv2.Rodrigues(np.array(vector).astype(np.float32))[0]
        vec = rot[:, 2]
        theta = np.arcsin(vec[1])
        phi = np.arctan2(vec[0], vec[2]) 
        if self.eye == "right":
            return rot, np.array([theta, -phi]).astype(np.float32)
        return rot, np.array([theta, phi]).astype(np.float32)
    
    def variance_of_laplacian(self, image):
        # compute the Laplacian of the image and then return the focus
        # measure, which is simply the variance of the Laplacian
        return cv2.Laplacian(image, cv2.CV_64F).var()

    def __len__(self):
        return len(self.labels_list)

    def __getitem__(self, index):
        
        good = 1
        
        if torch.is_tensor(index):
            index = index.tolist()
        
        img = self.image_list[index]
        pose= np.array(self.pose_list[index])
        pose, _ = self.convert_pose(pose)
        
        if self.eye == "right":
            ## flip image
            img = np.fliplr(img)
        

        pose = torch.tensor(pose)
        rms = img.std()
        fm = self.variance_of_laplacian(img)
        
        ## check if the image quality is bad
        if rms>75 or fm<200:
            good = 0

        img = torch.tensor(img.copy()).unsqueeze(-1)
        x = torch.tensor(self.labels_list[index][0])
        y = torch.tensor(self.labels_list[index][1])
        z = torch.tensor(self.labels_list[index][2])
        
        theta = torch.asin(-1*y)
        phi = torch.atan2(-1*x, -1*z) 
        if self.eye == "right":
            g  = torch.tensor([theta,-phi])
        else:
            g  = torch.tensor([theta, phi])
        
        return img, g, good, pose

## iterate over all the persons in the MPIIGaze Data  
for person in persons:
    
    print("Loading Person : {} Data".format(person))
    ## MPII dataset path
    data_path = "../../data/MPIIGaze/Data/Normalized/p" + person 
    
    ## Dump Path
    dump_root_path = "../../MPIIGaze/Segregated_Data/"
    os.makedirs(dump_root_path + "p" + person, exist_ok=True)
    dump_path = dump_root_path + "p" + person + "/"
    
    ## MPII Left Eye Image Dataset
    print("Loading Left Eye Data")
    dataset_left_eye = MPIIGazeSegregation(data_path, eye = 'left')
    
    ## MPII Right Eye Image Dataset
    print("Loading Right Eye Data")
    dataset_right_eye = MPIIGazeSegregation(data_path, eye = 'right')
    
    ## Concat both the datasets
    both_eyes_dataset = ConcatDataset([dataset_left_eye, dataset_right_eye])
    
    ## Segregation Based on IQ
    print ("Starting Data Segregation....")
    bad_dataset = []
    good_dataset = []
    for batch in tqdm(both_eyes_dataset):
        if batch[2] == 0:
            bad_dataset.append(batch)
        else:
            good_dataset.append(batch)
    
    ## Get the Dataloader     
    good_loader = torch.utils.data.DataLoader(good_dataset, batch_size=256, shuffle=True)
    bad_loader = torch.utils.data.DataLoader(bad_dataset, batch_size=256, shuffle=True)
    
    ## Save to H5 on disc
    save_on_disk(dump_path, person, good_loader, "good")
    save_on_disk(dump_path, person, bad_loader, "bad")
    
    
    
    


    
