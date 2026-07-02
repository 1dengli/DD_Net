import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import random
import pickle
import cv2
from Laplacian_Pyramid import laplacian_pyramid,reconstruct_image,reconstruct_image2



def bingji(a,b,c,d):
    t,k = a.shape
    out = np.zeros([t,k])
    for i in range(t):
        for j in range(k):
            if a[i,j]+ b[i,j]+ c[i,j] + d[i,j] >= 1:
                out[i,j] =1
            else:
                out[i,j] = 0
    return out

class LIDC_IDRI(Dataset):
    images = []
    labels = []
    series_uid = []

    def __init__(self, dataset_location, transform=None):
        self.transform = transform
        max_bytes = 2**31 - 1
        data = {}
        for file in os.listdir(dataset_location):
            filename = os.fsdecode(file)
            if '.pickle' in filename:
                print("Loading file", filename)
                file_path = dataset_location + filename
                bytes_in = bytearray(0)
                input_size = os.path.getsize(file_path)
                with open(file_path, 'rb') as f_in:
                    for _ in range(0, input_size, max_bytes):
                        bytes_in += f_in.read(max_bytes)
                new_data = pickle.loads(bytes_in)
                data.update(new_data)
        for key, value in data.items():
            self.images.append(value['image'].astype(np.float32))
            self.labels.append(value['masks'])
            self.series_uid.append(value['series_uid'])

        assert (len(self.images) == len(self.labels) == len(self.series_uid))

        for img in self.images:
            assert np.max(img) <= 1 and np.min(img) >= 0
        for label in self.labels:
            assert np.max(label) <= 1 and np.min(label) >= 0

        del new_data
        del data

    def __getitem__(self, index):
        image = np.expand_dims(self.images[index], axis=0)
        # image = ((self.images[index])*255).astype(np.uint8)
        # image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)/255
        # 高斯滤波降噪
        #Randomly select one of the four labels for this image
        #label = self.labels[index][random.randint(0,img_npy)].astype(float)
        label = bingji(self.labels[index][0].astype(float),self.labels[index][1].astype(float),self.labels[index][2].astype(float),self.labels[index][3].astype(float))

        gaussian = cv2.GaussianBlur((label)*255, (5, 5), 0)
        gaussian = gaussian.astype(np.uint8)
        # Canny算子
        Canny = cv2.Canny(gaussian, 0, 255)

        if self.transform is not None:
            image = self.transform(image)

        series_uid = self.series_uid[index]

        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        label_canny = torch.from_numpy(Canny/255)

        #Convert uint8 to float tensors
        image = image.type(torch.FloatTensor)
        label = label.type(torch.FloatTensor)
        label_canny = label_canny.type(torch.FloatTensor)

        return image, label, label_canny, series_uid

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    data=LIDC_IDRI(dataset_location='D:\\fjj_code\\resnet_seg\\data\\')