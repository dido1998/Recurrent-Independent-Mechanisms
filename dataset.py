"""Logic to interface with the dataset"""
from __future__ import print_function

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


class LoadDataset(Dataset):
    """Dataset class"""

    def __init__(self, mode, length=51, directory='/Data',
                 dataset="balls4mass64.h5"):
        self.length = length
        self.mode = mode
        self.directory = directory
        # datasets = ['/atari.h5', '/balls3curtain64.h5', '/balls4mass64.h5',
        #             '/balls678mass64.h5']

        hdf5_file = h5py.File(self.directory + "/" + dataset, 'r')

        if mode == "transfer":
            self.input_data = hdf5_file['test']
        else:
            self.input_data = hdf5_file[self.mode]
        self.input_data = np.array(self.input_data['features'])

    def __getitem__(self, index, out_list=('features', 'groups')):
        # ['collisions', 'events', 'features', 'groups', 'positions', 'velocities']
        # Currently (51 ,64, 64, 1)

        features = 1.0 * self.input_data[:self.length, index, :, :, :]
        # True, False label, conert to int
        # Convert to tensors
        L = self.input_data.shape[0]
        features = torch.tensor(features.reshape(L, 1, 64, 64))
        return features.float()

    def __len__(self):
        return int(np.shape(self.input_data)[1])





class LoadRGBDataset(Dataset):
     def __init__(self, mode, length=51, directory='/Data', dataset="balls4mass64.h5"):
         self.length = length
         self.mode   = mode
         self.directory = directory
         hdf5_file = h5py.File(self.directory + "/" + dataset, 'r')

         #if mode == "transfer":
         #   self.input_data = hdf5_file['test']
         #else:
         #   self.input_data = hdf5_file[self.mode]
         #self.input_data = np.array(self.input_data['features'])

         #datasets  = ['/atari.h5','/balls3curtain64.h5','/balls4mass64.h5','/balls678mass64.h5']
         hdf5_file = h5py.File(self.directory + "/" + dataset, 'r')
         #if mode != 'transfer':
         #    print('READING IN 4 DATASET')
         #    hdf5_file = h5py.File(self.directory+'/balls4mass64.h5', 'r')
         #else:
         #    print('READING IN 6-7-8 DATASET')
         #    hdf5_file = h5py.File(self.directory+'/balls678mass64.h5', 'r')

         if mode != 'transfer':
             self.input_data   = hdf5_file[self.mode]
         else:
             self.input_data   = hdf5_file['test']
         print(self.input_data)
         self.data_to_use = np.array(self.input_data['groups'])
         print("Done with RGB Convert")

     def __getitem__(self, index, out_list=('features', 'groups')):
         # ['collisions', 'events', 'features', 'groups', 'positions', 'velocities']
         # Currently (51 ,64, 64, 1)
         # print("In get item")
         # print('data_to_use shape: ',self.data_to_use.shape)
         # print('index is ',index)
         features = 1.0*self.data_to_use[:,index,:,:,:] # True, False label, conert to int
         #print(features.shape)

         colors = np.array([[228,26,28],[55,126,184],[77,175,74],[152,78,163],[255,127,0],[255,255,51]])/255.
         (Time, X_dim, Y_dim, Channels) = features.shape

         #print(self.data_to_use.shape)
         uniques = np.unique(features)[1:]
         uniques = uniques.astype(int)
         rc = [np.random.choice([0,1,2,3]) for _ in range(len(uniques))]

         self.data_to_use2 = np.zeros((Time, 3, X_dim, Y_dim))
         for t in range(Time):

             r_channel = np.zeros((64,64))
             g_channel = np.zeros((64,64))
             b_channel = np.zeros((64,64))
             # use four colours
             for ball in uniques:
                 self.data_to_use2[t,0,:,:] += ((features[t,:,:,0]==ball)*1.0)*colors[rc[ball-1]][0]
                 self.data_to_use2[t,1,:,:] += ((features[t,:,:,0]==ball)*1.0)*colors[rc[ball-1]][1]
                 self.data_to_use2[t,2,:,:] += ((features[t,:,:,0]==ball)*1.0)*colors[rc[ball-1]][2]
         features = self.data_to_use2
         features_no_noise = np.copy(features)
         features = torch.tensor(features)
         features_no_noise = torch.tensor(features_no_noise)
         #exit()
         #print(features.float().shape)
         return (features.float(),features_no_noise.float())

     def __len__(self):
         return int(np.shape(self.data_to_use)[1])

def get_dataloaders(args):
    """Method to return the train, test and transfer dataloaders"""

    modes = ["training", "test", "transfer"]
    dataset_names = [args.train_dataset, args.train_dataset, args.transfer_dataset]
    shuffle_list = [True, False, False]

    def _get_dataloader(mode, dataset_name, shuffle):
        dataset = LoadDataset(mode=mode,
                              length=args.sequence_length,
                              directory=args.directory,
                              dataset=dataset_name)
        return DataLoader(dataset, batch_size=args.batch_size,
                          shuffle=shuffle, num_workers=0)

    return [_get_dataloader(mode, dataset_name, shuffle) for
            mode, dataset_name, shuffle in zip(modes, dataset_names, shuffle_list)]

def get_rgb_dataloaders(args):
    """Method to return the train, test and transfer dataloaders"""

    modes = ["training", "test", "transfer"]
    dataset_names = [args.train_dataset, args.train_dataset, args.transfer_dataset]
    shuffle_list = [True, False, False]

    def _get_dataloader(mode, dataset_name, shuffle):
        dataset = LoadRGBDataset(mode=mode,
                              length=args.sequence_length,
                              directory=args.directory,
                              dataset=dataset_name)
        return DataLoader(dataset, batch_size=args.batch_size,
                          shuffle=shuffle, num_workers=0)

    return [_get_dataloader(mode, dataset_name, shuffle) for
            mode, dataset_name, shuffle in zip(modes, dataset_names, shuffle_list)]
