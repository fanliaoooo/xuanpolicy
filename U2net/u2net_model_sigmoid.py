import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

#训练集数据读取
with open('dataset.csv','r') as csvfile:
    reader = csv.reader(csvfile)
    flatten_data = list(reader)
with open('dataset_label.csv','r') as csvfile:
    reader = csv.reader(csvfile)
    label = list(reader)

input_data = np.array(flatten_data, dtype=np.float64)
inputs = input_data.reshape(input_data.shape[0],10,-1)
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
input_valid = input_valid.reshape(input_valid.shape[0],10,-1)
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
input_test = input_test.reshape(input_test.shape[0],10,-1)
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




# 定义神经网络模型
class REBNCONV(nn.Module):
    def __init__(self,in_ch=3,out_ch=3,dirate=1):
        super(REBNCONV,self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self,x):

        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout

def _upsample_like(src,tar):

    src = F.interpolate(src,size=tar.shape[2:],mode='bilinear')

    return src

class RSU2(nn.Module):#UNet04DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU2,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

class RSU2F(nn.Module):#UNet04FRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU2F,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=4)


        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3,hx2),1))
        hx1d = self.rebnconv1d(torch.cat((hx2d,hx1),1))

        return hx1d + hxin

class U2NETP(nn.Module):

    def __init__(self,in_ch=1,out_ch=1):
        super(U2NETP,self).__init__()

        self.stage1 = RSU2(in_ch,4,16)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2 = RSU2F(16,4,16)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage3 = RSU2F(16,4,16)

        # decoder
        self.stage2d = RSU2F(32,4,16)
        self.stage1d = RSU2(32,4,16)

        self.side1 = nn.Conv2d(16,out_ch,3,padding=1)
        self.side2 = nn.Conv2d(16,out_ch,3,padding=1)
        self.side3 = nn.Conv2d(16,out_ch,3,padding=1)

        self.outconv = nn.Conv2d(3*out_ch,out_ch,1)
        self.sigmoid = nn.Sigmoid()


    def forward(self,x):

        hx = x

        #stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        hx3up = _upsample_like(hx3,hx2)

        #decoder
        hx2d = self.stage2d(torch.cat((hx3up,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))

        #side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2,d1)

        d3 = self.side3(hx3)
        d3 = _upsample_like(d3,d1)

        d0 = self.outconv(torch.cat((d1,d2,d3),1))

        y2 = self.sigmoid(d0)
        out = y2.squeeze(1)
        out = out.permute(0, 2, 1)
        out = out.reshape(-1, out.size(-1))

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
model = U2NETP()
model.to('cuda:0')
# 定义损失函数和优化器
# criterion = ListMLE()  # 交叉熵损失函数，用于多分类问题
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 优化器
# scheduler = StepLR(optimizer, step_size=200, gamma = 0.5)


#pretrained_weights_path = '10v5__20000__cnn.pth'
#model.load_state_dict(torch.load(pretrained_weights_path))
#训练
train(model,input_data, label, criterion, optimizer,1000,input_data.size(0),4)

# 保存模型
# test(model,input_test,label_test)
test(model,input_data,label, criterion,1)

torch.save(model.state_dict(), 'target_allocation_model1.pth')

