import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(5, 10) # k : 5, n : 10
        self.linear2 = nn.Linear(10, 5) 

    def forward(self, x):
        import ipdb; ipdb.set_trace()
        x = F.relu(self.linear1(x))
        return F.relu(self.linear2(x))
    
class MyModule(torch.nn.Module):
    def __init__(self, k, n):
        super().__init__()
        self.linear1 = nn.Linear(k, n) # k : 5, n : 10
        self.linear2 = nn.Linear(n, k)
        self.act1 = nn.GELU()
        self.act2 = nn.Sigmoid()
        self.loss = torch.nn.MSELoss()
        
    def forward(self, input, label):
        output = self.linear1(input)
        output = self.act1(output)
        output = self.linear2(output)
        output = self.act2(output)
        loss = self.loss(output, label)
        return loss      
    
def nn_demo():
    '''
    1. 数据准备：输入数据 + lable 数据
    2. 网络结构的搭建：激活函数 + 损失函数 + 权重初始化；
    3. 优化器选择；
    4. 训练策略：学习率的控制 + 梯度清0 + 更新权重 + 正则化；
    '''
    
    model = MyModule(2, 3).cuda() # H2D --> 
    input = torch.tensor([5, 10]).reshape(1, 2).to(torch.float32).cuda()
    label = torch.tensor([0.01, 0.99]).reshape(1, 2).cuda()      
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    
    for i in range(100):
        # optimizer.zero_grad()
        model.zero_grad()
        loss = model(input, label)        
        loss.backward()
        optimizer.step()   
        print(loss)    
    
if __name__ == '__main__':
    nn_demo()

    
    
    
    