import json
from operator import itemgetter
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('hfl/rbt3')
rbt3 = AutoModel.from_pretrained('./checkpoint/rbt3')

class RankNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(768, 1)

    def forward(self, x, attentionMask):
        output = rbt3(input_ids=x, attention_mask=attentionMask)
        output = self.linear(output.pooler_output)
        return output

model = RankNet()
model.load_state_dict(torch.load('./checkpoint/RankNet'))
model.eval()

rank = []

with open('./dataset/validation/女神異聞錄3.json', encoding='utf-8') as jsonFile:
    reviews = json.load(jsonFile)

for review in reviews:
    input_review = tokenizer(review, return_tensors='pt', padding='max_length', max_length=128, truncation=True)
    score = model(input_review.input_ids, input_review.attention_mask)
    rank.append([review, score])

rank.sort(key=itemgetter(1), reverse=True)
for i in rank:
    print(i[0])
    print('=============================')
