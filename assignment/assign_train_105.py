import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv

#训练集数据读取
with open('dataset105.csv','r') as csvfile:
    reader = csv.reader(csvfile)
    flatten_data = list(reader)
with open('dataset_label105.csv','r') as csvfile:
    reader = csv.reader(csvfile)
    label = list(reader)

input_data = np.array(flatten_data, dtype=np.float64)
inputs = input_data.reshape(20000,1,10,5)
label = np.array(label, dtype=np.float32)

input_data = torch.tensor(inputs, device='cuda:0')
label = torch.tensor(label, device='cuda:0')
label = label.to(torch.int64)

#验证集数据读取
with open('validset105.csv','r') as csvfile:
    reader = csv.reader(csvfile)
    flatten_validdata = list(reader)
with open('validset_label105.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    labelvalid = list(reader)

input_valid = np.array(flatten_validdata, dtype=np.float64)
input_valid = input_valid.reshape(5000,1,10,5)
label_valid = np.array(labelvalid, dtype=np.float32)

input_valid = torch.tensor(input_valid, device='cuda:0')

label_valid = torch.tensor(label_valid, device='cuda:0')
label_valid= label_valid.to(torch.int64)


#测试集数据读取
with open('testset105.csv','r') as csvfile:
    reader = csv.reader(csvfile)
    flatten_testdata = list(reader)
with open('testset_label105.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    labeltest = list(reader)

input_test = np.array(flatten_testdata, dtype=np.float64)
input_test = input_test.reshape(1000,1,10,5)
label_test = np.array(labeltest, dtype=np.float32)

input_test = torch.tensor(input_test, device='cuda:0')
label_test = torch.tensor(label_test, device='cuda:0')
label_test = label_test.to(torch.int64)



# 定义神经网络模型
class TargetAllocationModel(nn.Module):
    def __init__(self):
        super(TargetAllocationModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels= 1, kernel_size=3, stride = 1, padding = 1, bias=False)
        self.bn1 = nn.BatchNorm2d(1)
        self.relu = nn.Sigmoid()
        self.fc1 = nn.Linear(5, 256)
        # self.bn = nn.BatchNorm1d(5)
        self.relu = nn.Sigmoid()

        # self.dropout = nn.Dropout(p=0.2)

        self.fc2 = nn.Linear(256, 256)
        self.relu2 = nn.Sigmoid()

        # self.fc3 = nn.Linear(256, 64)
        # self.relu3 = nn.Sigmoid()
        # self.fc4 = nn.Linear(256, 128)
        # self.relu4 = nn.Sigmoid()

        self.fc4 = nn.Linear(256,5)
        self.softmax = nn.LogSoftmax(dim=2)  # 在输出层使用 softmax 激活函数

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc1(out)
        # out = self.bn(out)
        out = self.relu(out)
        out = out.squeeze(1)

        # out = self.dropout(out)

        out = self.fc2(out)
        out = self.relu2(out)

        # out = self.fc3(out)
        # out = self.relu3(out)

        out = self.fc4(out)
        out = self.softmax(out)

        out = out.permute(0, 2, 1)
        return out


def train(model,input_loader, labels,criterion, optimizer, num_epochs,num_samples,batch_size):
    model.train()
    for epoch in range(num_epochs):
        indices = torch.randperm(input_loader.size(0))
        shuffled_data = input_loader[indices]
        shuffled_label = labels[indices]
        running_loss = 0.0

        for i in range(0, num_samples, batch_size):
            inputs = shuffled_data[i:i + batch_size]
            targets = shuffled_label[i:i + batch_size]
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
        print("*****************************************")
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss/(num_samples/batch_size):.4f}')
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
                label = test_label[i]
                label = label.unsqueeze(0)
                outputs = model(inputs.float())
                loss = criterion(outputs, label)
                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += label.size(1)
                correct += (predicted == label).sum().item()
        accuracy = correct / total
        print(f'Loss_on_valid_set: {running_loss/5000:.4f}')
        print(f'Accuracy on valid set: {accuracy}')
        print("*****************************************")

    elif mode == 1: # 1为测试数据
        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(test_input.size(0)):
                inputs = test_input[i]
                inputs = inputs.unsqueeze(0)
                label = test_label[i]
                outputs = model(inputs.float())

                _, predicted = torch.max(outputs, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
        accuracy = correct / total
        print(f'Accuracy on test set: {accuracy}')




# 创建模型实例
model = TargetAllocationModel()
model.to('cuda:0')
# 定义损失函数和优化器
criterion = nn.NLLLoss()  # 交叉熵损失函数，用于多分类问题
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 优化器


#pretrained_weights_path = '10v5__20000__cnn.pth'
#model.load_state_dict(torch.load(pretrained_weights_path))
#训练
train(model,input_data, label, criterion, optimizer,2000,20000,80)

# 保存模型
# test(model,input_test,label_test)
test(model,input_test,label_test, criterion,1)

torch.save(model.state_dict(), 'target_allocation_model1.pth')

