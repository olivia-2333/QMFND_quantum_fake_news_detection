
import torch
from datasets import FakeNewsDataset, create_mini_batch
from torch.utils.data import DataLoader
from torch import nn
import qnn

BATCH_SIZE = 56

from transformers import AutoTokenizer, XLNetForSequenceClassification
from IPython.display import display, clear_output

proxy={'http':'http://127.0.0.1:7890','https':'http://127.0.0.1:7890'}
MODEL_NAME = 'xlnet-base-cased'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,proxies=proxy)

cla_model = XLNetForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=4,proxies=proxy)

qcnn=qnn.QCNN(4)

data_path='/media/zll/BLACK1/QML/QCNN_FND/data/'
trainset = FakeNewsDataset('train', tokenizer=tokenizer, path=data_path,datasets='gossip')
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, collate_fn=create_mini_batch)


from sklearn.metrics import accuracy_score
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print('device:', device)
cla_model.to(device)
qcnn.to(device)

cla_model.train()
qcnn.train()


optimizer = torch.optim.AdamW([
    {'params': cla_model.parameters()},
    {'params': qcnn.parameters()}
], lr=1e-3)

loss_func = nn.CrossEntropyLoss()

NUM_EPOCHS = 50
print(f"data length:{trainset.len}")
for epoch in range(NUM_EPOCHS):
    train_loss = 0.0
    train_acc = 0.0
    
    loop = tqdm(trainloader)
    for batch_idx, data in enumerate(loop):


        tokens_tensors, segments_tensors, masks_tensors, labels = [t.to(device) for t in data]
        
        optimizer.zero_grad()
        q_input = cla_model(input_ids=tokens_tensors,
                               token_type_ids=segments_tensors,
                               attention_mask=masks_tensors)[0]
        
        outputs=qcnn(q_input)

        loss = loss_func(outputs,labels)

        loss.backward()
        optimizer.step()
        
        pred = torch.argmax(outputs, dim=1)
        train_acc = accuracy_score(pred.cpu().tolist(), labels.cpu().tolist())
        
        train_loss += loss.item()
        
        loop.set_description(f'Epoch [{epoch + 1}/{NUM_EPOCHS}]')
        loop.set_postfix(acc=train_acc, loss=train_loss/(batch_idx+1))

    torch.save(cla_model, 'result/xlnet_politi_tx.pth')
    torch.save(qcnn.state_dict(), 'result/qcnn_politi_tx.pth')



torch.save(cla_model, 'result/xlnet_politi_tx.pth')
torch.save(qcnn.state_dict(), 'result/qcnn_politi_tx.pth')
