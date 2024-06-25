from __future__ import print_function
import os.path as osp
from torch.utils.data import Dataset
from tqdm import tqdm
from .utils import HeadPipeline

class headdataset(Dataset):
    def __init__(self, data_dir, ann_path, test_mode=False,data_aug=False,input_size=80):
        super().__init__()

        self.data_dir = data_dir
        self.ann_path = ann_path
        self.test_mode = test_mode
        self.data_aug = data_aug
        self.input_size = input_size

        self.get_data()

    def get_data(self):
        """Get data from a provided annotation file.
        """
        self.data_items = []
        self.label_items = []

        with open(self.ann_path, 'r') as f:
            lines = f.readlines()
        for line in tqdm(lines):
            path, label = line.rstrip().split()
            self.data_items.append(path)
            self.label_items.append(int(label))
        if len(self.data_items) == 0:
            raise (RuntimeError('Found 0 files.'))
        f.close()
        print('-----Load image_path and label is successful----- ')

    def prepare(self, idx):
        # load image and pre-process (pipeline)
        path = self.data_items[idx]
        path = osp.join(self.data_dir,path)
        image = HeadPipeline(path, self.test_mode,self.data_aug,self.input_size)
        label = self.label_items[idx]
        return image, label

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        return self.prepare(idx)
