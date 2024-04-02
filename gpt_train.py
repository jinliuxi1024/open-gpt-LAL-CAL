from gpt_tokenizer import Tokenizer
import torch
import numpy as np
import matplotlib.pyplot as plt
from gpt_model import GPTmodel
import os
class Trainer:
    def __init__(self,config):
        self.tokenizer = Tokenizer()
        self.batch_size = config.batch_size
        self.block_size = config.block_size
        self.device = config.device
        self.model = config.model
        self.train_dir = config.train_dir
        #读取训练数据,如果是文件夹，读取文件夹下所有文件
        if os.path.isdir(self.train_dir):
            self.text = ''
            for file in os.listdir(self.train_dir):
                with open(os.path.join(self.train_dir, file), 'r', encoding='utf-8') as f:
                    self.text += f.read()
        else:
            with open(self.train_dir, 'r', encoding='utf-8') as f:
                self.text = f.read()
        self.text = self.tokenizer.encode(self.text)
        self.text_len = len(self.text)
        #划分训练集和验证集
        self.train_data_x = self.text[:int(self.text_len*0.9)]
        self.train_data_y = self.text[1:int(self.text_len*0.9)+1]
        self.valid_data_x = self.text[int(self.text_len*0.9):-1]
        self.valid_data_y = self.text[1+int(self.text_len*0.9):]

    def run(self,config):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        # 实时数据可视化
        train_losses = []
        valid_losses = []
        for epoch in range(config.epochs):
            #随机取样
            idx = np.random.randint(0, len(self.train_data_x) - self.block_size, self.batch_size)
            train_x =[]
            train_y = []
            for i in range(self.batch_size):
                train_x.append(self.train_data_x[idx[i]:idx[i]+self.block_size])
                train_y.append(self.train_data_y[idx[i]:idx[i]+self.block_size])
            train_x = torch.tensor(train_x, dtype=torch.long).to(self.device)
            train_y = torch.tensor(train_y, dtype=torch.long).to(self.device)
            output, loss = self.model(train_x, train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                #验证集随机取一份
                idx = np.random.randint(0, len(self.valid_data_x) - self.block_size, 1)
                valid_x = self.valid_data_x[idx[0]:idx[0] + self.block_size]
                valid_y = self.valid_data_y[idx[0]:idx[0] + self.block_size]
                valid_x = torch.tensor(valid_x, dtype=torch.long).unsqueeze(0).to(self.device)
                valid_y = torch.tensor(valid_y, dtype=torch.long).unsqueeze(0).to(self.device)
                _, valid_loss = self.model(valid_x, valid_y)
                valid_losses.append(valid_loss.item())
                train_losses.append(loss.item())
                #绘制训练集和测试集的loss
                #输出测试集的loss
                print('epoch:', epoch, 'train_loss:', loss.item(), 'valid_loss:', valid_loss.item())
        #保存模型
        torch.save(self.model.state_dict(), 'model.pth')
        print('model has been saved')



