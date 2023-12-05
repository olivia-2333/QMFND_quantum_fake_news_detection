from torch.utils.data import Dataset    
import torch
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from torchvision import transforms

class FakeNewsDataset(Dataset):
    def __init__(self, mode,datasets, tokenize,path,img_path):

        assert mode in ['train', 'test']
        self.mode=mode
        self.img_path=img_path

        self.df = pd.read_csv(path + datasets+'_'+mode + '.csv').fillna('')
        self.len = len(self.df)
        self.tokenizer = tokenize
        
    def __getitem__(self, idx):

        statement=self.df.iloc[idx]['content']
        label=self.df.iloc[idx]['label']
        img=self.df.iloc[idx]['image']
        label_tensor = torch.tensor(label)
            
        word_pieces = ['[CLS]']
        statement = self.tokenizer.tokenize(statement)
        if len(statement) > 100:
            statement = statement[:100]
        word_pieces += statement + ['[SEP]']
        len_st = len(word_pieces)
            
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)
            
        segments_tensor = torch.tensor([0] * len_st, dtype=torch.long)
        image=Image.open(self.img_path+img)
        #将图片转换为tensor
        image_tensor=transforms.ToTensor()(image)

        return (tokens_tensor, segments_tensor, image_tensor,label_tensor)
        
    def __len__(self):
        return self.len


def create_mini_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]
    imgs_tensors=[s[2] for s in samples]
    imgs_tensors=torch.stack(imgs_tensors)

    if samples[0][3] is not None:

        label_ids = torch.stack([s[3] for s in samples])
    else:
        label_ids = None
        

    tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, batch_first=True)
    
    masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)

    masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0,1)
    
    return tokens_tensors, segments_tensors, masks_tensors, imgs_tensors,label_ids