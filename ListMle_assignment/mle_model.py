import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
from torch.optim.lr_scheduler import StepLR

#训练集数据读取
with open('dataset.csv','r') as csvfile:
    reader = csv.reader(csvfile)
    flatten_data = list(reader)
with open('dataset_label.csv','r') as csvfile:
    reader = csv.reader(csvfile)
    label = list(reader)

input_data = np.array(flatten_data, dtype=np.float64)
inputs = input_data.reshape(input_data.shape[0],5,-1)
#预处理数据

min_train_set = np.amin(inputs, axis=(1,2))
min_values = min_train_set.reshape(inputs.shape[0],1,1)
inputs = inputs - min_values
max_train_set = np.amax(inputs,axis=(1,2))
max_values = max_train_set.reshape(inputs.shape[0],1,1)
inputs = inputs/max_values

# mean = np.mean(inputs,axis = 1)
# std = np.std(inputs, axis = 1)
# inputs = (inputs - mean[:,np.newaxis,:])/std[:,np.newaxis,:]

label = np.array(label, dtype=np.float32)


input_data = torch.tensor(inputs, device='cuda:0')
label = torch.tensor(label, device='cuda:0')
label = label.to(torch.float32)

#验证集数据读取
with open('validset.csv','r') as csvfile:
    reader = csv.reader(csvfile)
    flatten_validdata = list(reader)
with open('validset_label.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    labelvalid = list(reader)

input_valid = np.array(flatten_validdata, dtype=np.float64)
input_valid = input_valid.reshape(input_valid.shape[0],5,-1)
label_valid = np.array(labelvalid, dtype=np.float32)
##预处理数据

min_valid_set = np.amin(input_valid, axis=(1,2))
min_valid = min_valid_set.reshape(input_valid.shape[0],1,1)
input_valid = input_valid- min_valid
max_valid_set = np.amax(input_valid,axis=(1,2))
max_valid = max_valid_set.reshape(input_valid.shape[0],1,1)
input_valid = input_valid/max_valid
# mean = np.mean(input_valid,axis = 1)
# std = np.std(input_valid, axis = 1)
# input_valid = (input_valid - mean[:,np.newaxis,:])/std[:,np.newaxis,:]

input_valid = torch.tensor(input_valid, device='cuda:0')

label_valid = torch.tensor(label_valid, device='cuda:0')
label_valid= label_valid.to(torch.float32)


#测试集数据读取
with open('testset.csv','r') as csvfile:
    reader = csv.reader(csvfile)
    flatten_testdata = list(reader)
with open('testset_label.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    labeltest = list(reader)

input_test = np.array(flatten_testdata, dtype=np.float64)
input_test = input_test.reshape(input_test.shape[0],5,-1)
label_test = np.array(labeltest, dtype=np.float32)
##预处理数据

min_test_set = np.amin(input_test, axis=(1,2))
min_test = min_test_set.reshape(input_test.shape[0],1,1)
input_test = input_test- min_test
max_test_set = np.amax(input_test,axis=(1,2))
max_test= max_test_set.reshape(input_test.shape[0],1,1)
input_test = input_test/max_test
# mean = np.mean(input_test,axis = 1)
# std = np.std(input_test, axis = 1)
# input_test = (input_test - mean[:,np.newaxis,:])/std[:,np.newaxis,:]

input_test = torch.tensor(input_test, device='cuda:0')
label_test = torch.tensor(label_test, device='cuda:0')
label_test = label_test.to(torch.float32)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.residual = nn.Sequential(
                            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3,padding=1, bias=False),
                            nn.BatchNorm2d(in_channels // 2),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(in_channels // 2, out_channels, kernel_size=3,padding=1, bias=False),
                            nn.BatchNorm2d(out_channels)
                        )
    def forward(self, x):
        out = self.residual(x)
        return out


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=1)
        self.residual = ResidualBlock(in_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out1 = self.skip(x)
        out2 = self.residual(x)
        out = self.relu(out1 + out2)
        return out

# 定义神经网络模型
class TargetAllocationModel(nn.Module):
    def __init__(self):
        super(TargetAllocationModel, self).__init__()

        self.fc_1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3,padding=1),
            nn.ReLU(),
            Encoder(128, 32),
            Encoder(32, 8)
        )
        self.fc_2 = nn.Sequential(
            nn.Conv2d(8, 4, 1),
            nn.ReLU(),
            nn.Conv2d(4, 1, 1)
        )
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.fc_1(x)
        y1 = self.fc_2(x)
        y2 = y1.squeeze(1)

        out = self.softmax(y2)
        out = out.permute(0,2,1)
        out = out.reshape(-1,out.size(-1))

        return out


def train(model,input_loader, labels,criterion, optimizer,num_epochs,num_samples,batch_size):
    for epoch in range(num_epochs):
        model.train()
        indices = torch.randperm(input_loader.size(0))
        shuffled_data = input_loader[indices]
        shuffled_label = labels[indices]
        running_loss = 0.0

        for i in range(0, num_samples, batch_size):
            inputs = shuffled_data[i:i + batch_size]
            inputs = inputs.unsqueeze(1)
            targets = shuffled_label[i:i + batch_size]
            targets = targets.reshape(-1, inputs.size(2), inputs.size(3))
            targets = targets.permute(0, 2, 1)
            targets = targets.reshape(-1, targets.size(-1))

            # targets = targets.view(-1)
            #
            # inputs = input_data[i:i+batch_size]
            # targets = labels[i:i+batch_size]

            # 向前传播
            outputs = model(inputs.float())  # 需要将数据转换为 float 类型

            # 计算损失
            loss = criterion(outputs, targets)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # scheduler.step()
        print("*****************************************")
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss/(num_samples/batch_size):.4f}')
        # if not epoch%50:
        test(model,input_valid,label_valid,criterion,0)

def test(model, test_input, test_label,criterion,mode):
    model.eval()
    if mode == 0: #0为验证数据
        correct = 0
        total = 0
        running_loss = 0.0
        with torch.no_grad():
            for i in range(test_input.size(0)):
                inputs = test_input[i]
                inputs = inputs.unsqueeze(0)
                inputs = inputs.unsqueeze(0)
                label = test_label[i]
                label = label.reshape(-1, inputs.size(3))
                label = label.T
                # label = label.unsqueeze(0)
                outputs = model(inputs.float())
                loss = criterion(outputs, label)
                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += label.size(0)
                _, label_idx = torch.max(label, 1)
                correct += (predicted == label_idx).sum().item()
        accuracy = correct / total
        print(f'Loss_on_valid_set: {running_loss/4000:.4f}')
        print(f'Accuracy on valid set: {accuracy}')
        print("*****************************************")

    elif mode == 1: # 1为测试数据
        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(test_input.size(0)):
                inputs = test_input[i]
                inputs = inputs.unsqueeze(0)
                inputs = inputs.unsqueeze(0)
                label = test_label[i]
                label = label.reshape(-1, inputs.size(3))
                label = label.T
                outputs = model(inputs.float())

                _, predicted = torch.max(outputs, 1)
                total += label.size(0)
                _, label_idx = torch.max(label, 1)

                # sort_idx = torch.argsort(outputs, dim=1)
                # predicted = torch.argsort(sort_idx,dim=1)
                # total += label.size(0)
                correct += (predicted == label_idx).sum().item()
        accuracy = correct / total
        print(f'Accuracy on test set: {accuracy}')

class ListMLE(nn.Module):
    def __init__(self):
        super(ListMLE, self).__init__()
    def forward(self,outputs,labels):
        scores = torch.zeros_like(outputs)
        idx = torch.argsort(labels,dim=1)
        for t in range(scores.size(0)):
            scores[t] = torch.logcumsumexp(outputs[t,idx[t]],dim=0)
        loss = torch.mean(scores-outputs)
        return loss


# 创建模型实例
model = TargetAllocationModel()
model.to('cuda:0')
# 定义损失函数和优化器
# criterion = ListMLE()  # 交叉熵损失函数，用于多分类问题
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Adam 优化器
# scheduler = StepLR(optimizer, step_size=200, gamma = 0.5)


#pretrained_weights_path = '10v5__20000__cnn.pth'
#model.load_state_dict(torch.load(pretrained_weights_path))
#训练
train(model,input_data, label, criterion, optimizer,1000,input_data.size(0),4)

# 保存模型
# test(model,input_test,label_test)
test(model,input_data,label, criterion,1)

torch.save(model.state_dict(), 'target_allocation_model1.pth')

