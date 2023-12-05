from torch.utils.data import Dataset    
import torch
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from torchvision import transforms

class FakeNewsDataset(Dataset):
    def __init__(self, mode,datasets,path,img_path):

        assert mode in ['train', 'test']
        self.mode=mode
        self.img_path=img_path

        self.df = pd.read_csv(path + datasets+'_'+mode + '.csv').fillna('')
        self.len = len(self.df)
        
    def __getitem__(self, idx):

        label=self.df.iloc[idx]['label']
        img=self.df.iloc[idx]['image']
        label_tensor = torch.tensor(label)
            
        image=Image.open(self.img_path+img)

        image_tensor=transforms.ToTensor()(image)

        return (image_tensor,label_tensor)
        
    def __len__(self):
        return self.len

