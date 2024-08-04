import os
import json
import itertools
import numpy
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

# dataset

tokenizer = AutoTokenizer.from_pretrained('hfl/rbt3')

def order2postivePairs(order):
    pairs = [numpy.stack(combination) for combination in itertools.combinations(order, 2)]
    return pairs

pathToJson = './dataset/training/'
trainData = []
trainDataAttentionMask = []
for fileName in [file for file in os.listdir(pathToJson) if file.endswith('.json')]:
    with open(pathToJson + fileName, encoding='utf-8') as jsonFile:
        reviews = [review[1] for review in json.load(jsonFile)]
        reviews = tokenizer(reviews, return_tensors='pt', padding='max_length', max_length=128, truncation=True)
        trainData.extend(order2postivePairs(reviews.input_ids))
        trainDataAttentionMask.extend(order2postivePairs(reviews.attention_mask))

trainData = numpy.array(trainData)
trainDataAttentionMask = numpy.array(trainDataAttentionMask)

class PostivePairsDataset(Dataset):
    def __init__(self, trainData, trainDataAttentionMask):
        self.trainData = trainData
        self.trainDataAttentionMask = trainDataAttentionMask

    def __len__(self):
        return len(self.trainData)

    def __getitem__(self, idx):
        postivePair = torch.tensor(self.trainData[idx], dtype=torch.float32)
        attentionMask = torch.tensor(self.trainDataAttentionMask[idx], dtype=torch.float32)
        postiveLabel = torch.tensor([1], dtype=torch.float32)
        return postivePair, attentionMask, postiveLabel

batchSize = 128
dataloader = DataLoader(PostivePairsDataset(trainData, trainDataAttentionMask), batch_size = batchSize, shuffle = True)

# model

device = None
def getDevice():
    global device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device

rbt3 = AutoModel.from_pretrained('hfl/rbt3')
if getDevice() == 'cuda':
    rbt3 = rbt3.cuda()

class RankNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(768, 1)

    def forward(self, x, attentionMask):
        output = rbt3(input_ids=x, attention_mask=attentionMask)
        output = self.linear(output.pooler_output)
        return output

model = RankNet().to(getDevice())

# train

def train(dataloader, model, lossFunction, optimizer):
    model.train()
    currentCount = 0
    counter = 0
    for batch, (X, attentionMask, y) in enumerate(dataloader):
        counter = counter + 1
        X, attentionMask, y = X.to(getDevice()).long(), attentionMask.to(getDevice()).long(), y.to(getDevice())
        optimizer.zero_grad()
        X1 = X[:, 0]
        X2 = X[:, 1]
        score1 = model(X1, attentionMask[:, 0])
        score2 = model(X2, attentionMask[:, 1])
        out = score1 - score2
        loss = lossFunction(out, y)

        loss.backward()
        optimizer.step()

        postive = (out > 0).sum().item()
        currentCount = currentCount + postive
        if counter % 100 == 0:
            print(batch)
            print(postive / batchSize)

    datasetSize = len(dataloader.dataset)
    currentRate = currentCount / datasetSize
    return currentRate

lossFunction = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr = 2e-5)
epoch = 2
for i in range(epoch):
    trainCurrentRate = train(dataloader, model, lossFunction, optimizer)
    print(f'epoch{i} trainCurrentRate: {trainCurrentRate:.2f}')

for param in rbt3.parameters():
    param.data = param.data.contiguous()
rbt3.save_pretrained("./checkpoint/rbt3", from_pt=True)

for param in model.parameters():
    param.data = param.data.contiguous()
torch.save(model.state_dict(), './checkpoint/RankNet')
