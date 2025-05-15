import os

import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose

import transforms as t

from scipy.stats import pearsonr


class FWIDataset(Dataset):
    def __init__(self, anno, preload=True, sample_ratio=1, file_size=1,
                 transform_data1=None, transform_data2=None,
                 transform_label1=None, transform_label2=None):

        if not os.path.exists(anno):
            print(f'Annotation file {anno} does not exists')
        self.preload = preload
        self.sample_ratio = sample_ratio
        self.file_size = file_size

        self.transform_data1 = transform_data1
        self.transform_data2 = transform_data2
        self.transform_label1 = transform_label1
        self.transform_label2 = transform_label2
        with open(anno, 'r') as f:
            self.batches = f.readlines()
        if preload:
            self.data_y_list, self.data_x_list, self.label1_list,  self.label2_list = [], [], [], []
            for batch in self.batches:
                data_y, data_x, label1, label2 = self.load_every(batch)
                self.data_y_list.append(data_y)
                self.data_x_list.append(data_x)
                if label1 is not None:
                    self.label1_list.append(label1) 
                if label2 is not None:
                    self.label2_list.append(label2)

            # print the max and min values of the data and labels
            print(f"Data-y max: {np.max(data_y)}", f"Data-y min: {np.min(data_y)}")
            print(f"Data-x max: {np.max(data_x)}", f"Data-x min: {np.min(data_x)}")
            print(f"Label1 max: {np.max(label1)}", f"Label1 min: {np.min(label1)}")
            print(f"Label2 max: {np.max(label2)}", f"Label2 min: {np.min(label2)}")

    # Load from one line
    def load_every(self, batch):
        batch = batch.split('\t')
        data_y_path = batch[0]
        data_x_path = batch[1]

        data_y = np.load(data_y_path)[:, :, ::self.sample_ratio, :]
        data_y = data_y.astype('float32')
        data_x = np.load(data_x_path)[:, :, ::self.sample_ratio, :]
        data_x = data_x.astype('float32')

        # if len(batch) > 1:
        if len(batch) > 2:
            label1_path = batch[2][:-1]
            label1 = np.load(label1_path)
            label1 = label1.astype('float32')

            label2_path = batch[3][:-1]
            label2 = np.load(label2_path)
            label2 = label2.astype('float32')

        else:
            label1, label2 = None, None

        return data_y, data_x, label1, label2

    def __getitem__(self, idx):
        batch_idx, sample_idx = idx // self.file_size, idx % self.file_size
        if self.preload:
            data_y_arr = self.data_y_list[batch_idx]
            data_x_arr = self.data_x_list[batch_idx]
            if self.file_size == 1:
                # Retain the dummy dimension.
                data_y = data_y_arr  # shape: (1, T, C, H, W)
                data_x = data_x_arr  # shape: (1, T, C, H, W)
                label1 = self.label1_list[batch_idx] if len(self.label1_list) != 0 else None
                label2 = self.label2_list[batch_idx] if len(self.label2_list) != 0 else None
                # If labels are 4D, add a dummy dimension to make them 5D.
                if label1 is not None and label1.ndim == 4:
                    label1 = label1[None, ...]
                if label2 is not None and label2.ndim == 4:
                    label2 = label2[None, ...]
            else:
                # For a single sample from a file with multiple time steps:
                data_y = data_y_arr[0][sample_idx]  # shape: (C, H, W)
                data_x = data_x_arr[0][sample_idx]    # shape: (C, H, W)
                # Add a dummy batch dimension:
                data_y = data_y[None, ...]  # shape becomes (1, C, H, W)
                data_x = data_x[None, ...]  # shape becomes (1, C, H, W)
                if len(self.label1_list) != 0:
                    label1_arr = self.label1_list[batch_idx]
                    if label1_arr.ndim == 5:
                        label1_arr = label1_arr.squeeze(0)  # shape: (T, C, H, W)
                    label1 = label1_arr[sample_idx]
                    label1 = label1[None, ...]  # shape: (1, C, H, W)
                else:
                    label1 = None
                if len(self.label2_list) != 0:
                    label2_arr = self.label2_list[batch_idx]
                    if label2_arr.ndim == 5:
                        label2_arr = label2_arr.squeeze(0)
                    label2 = label2_arr[sample_idx]
                    label2 = label2[None, ...]
                else:
                    label2 = None
        else:
            data_y, data_x, label = self.load_every(self.batches[batch_idx])
            if self.file_size == 1:
                data_y = data_y
                data_x = data_x
                label1 = label1 if label is not None else None
                label2 = label2 if label is not None else None
                if label1 is not None and label1.ndim == 4:
                    label1 = label1[None, ...]
                if label2 is not None and label2.ndim == 4:
                    label2 = label2[None, ...]
            else:
                data_y = data_y[0][sample_idx]
                data_x = data_x[0][sample_idx]
                label1 = label[0][sample_idx] if label is not None else None
                label2 = label[0][sample_idx] if label is not None else None
                if label1 is not None and label1.ndim == 3:
                    label1 = label1[None, ...]
                if label2 is not None and label2.ndim == 3:
                    label2 = label2[None, ...]

        if self.transform_data1:
            data_y = self.transform_data1(data_y)
        if self.transform_data2:
            data_x = self.transform_data2(data_x)
        if self.transform_label1:
            label1 = self.transform_label1(label1)
        if self.transform_label2:
            label2 = self.transform_label2(label2)

        return data_y, data_x, label1, label2 

    def __len__(self):
        return len(self.batches) * self.file_size


if __name__ == '__main__':
    transform_data1 = Compose([
        t.LogTransform(k=1),
        t.MinMaxNormalize(t.log_transform(-61, k=1), t.log_transform(120, k=1))
    ])
    transform_data2 = Compose([
        t.LogTransform(k=1),
        t.MinMaxNormalize(t.log_transform(-61, k=1), t.log_transform(120, k=1))
    ])
    transform_label1 = Compose([
        t.MinMaxNormalize(1950, 3930)
    ])
    transform_label2 = Compose([
        t.MinMaxNormalize(1950, 3930)
    ])


    dataset = FWIDataset('relevant_files/temp.txt',
                         transform_data1=transform_data1,
                         transform_data2=transform_data2,
                         transform_label1=transform_label1,
                         transform_label2=transform_label2, 
                         file_size=1)

    data_y, data_x, label1, label2 = dataset[0]
    print("data_y shape:", data_y.shape)
    print("data_x shape:", data_x.shape)
    print("label1 shape:", label1.shape if label1 is not None else None)
    print("label2 shape:", label2.shape if label2 is not None else None)