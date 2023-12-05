from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
import qnn
from datasets_with_img import FakeNewsDataset, create_mini_batch

qbits=8
xlnet_path='xlnet_model_gossip.pth'
data_path='data/'
img_path='img/'
datasets='gossip'
mode='train'
BATCH_SIZE = 16

from transformers import AutoTokenizer, XLNetForSequenceClassification
from IPython.display import display, clear_output

proxy={'http':'http://127.0.0.1:7890','https':'http://127.0.0.1:7890'}
MODEL_NAME = 'xlnet-base-cased'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,proxies=proxy)
NUM_LABELS = 4

xlnet = XLNetForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS,proxies=proxy)


vgg = models.vgg19(pretrained=True,progress=True)
vgg.classifier[6] = nn.Linear(4096, qbits//2,bias=True)

qcnn=qnn.QCNN(qbits)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print('device:', device)
xlnet.to(device)
qcnn.to(device)
vgg.to(device)

xlnet.train()
qcnn.train()
vgg.train()



optimizer = torch.optim.AdamW([
    {'params': xlnet.parameters()},
    {'params': qcnn.parameters()},
    {'params': vgg.parameters()}
], lr=1e-3)

loss_func = nn.CrossEntropyLoss()
loss_v=[]
acc_v=[]
NUM_EPOCHS = 10
if mode=='train':
    trainset=FakeNewsDataset(mode, datasets=datasets,tokenize=tokenizer, path=data_path, img_path=img_path)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, collate_fn=create_mini_batch)
    print(f"data length:{trainset.len}")
    for epoch in range(NUM_EPOCHS):
        train_loss = 0.0
        train_acc = 0.0
        
        loop = tqdm(trainloader)
        for batch_idx, data in enumerate(loop):
  
            
            tokens_tensors, segments_tensors, masks_tensors, imgs_tensors,labels = [t.to(device) for t in data]
            
            optimizer.zero_grad()
            q_tx_input = xlnet(input_ids=tokens_tensors,
                                token_type_ids=segments_tensors,
                                attention_mask=masks_tensors)[0]
            q_img_input=vgg(imgs_tensors)
            q_input=torch.cat((q_tx_input,q_img_input),dim=1)
            outputs=qcnn(q_input)

            loss = loss_func(outputs,labels)
            # loss = outputs[0]
            loss.backward()
            optimizer.step()
            
            pred = torch.argmax(outputs, dim=1)
            train_acc = accuracy_score(pred.cpu().tolist(), labels.cpu().tolist())
            
            train_loss += loss.item()
            
            loop.set_description(f'Epoch [{epoch + 1}/{NUM_EPOCHS}]')
            loop.set_postfix(acc=train_acc, loss=train_loss/(batch_idx+1))
            loss_v.append(train_loss/(batch_idx+1))
            acc_v.append(train_acc)
        
        torch.save(loss_v,'loss_v.pth')
        torch.save(acc_v,'acc_v.pth')