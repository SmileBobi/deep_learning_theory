import torch

def reshape_demo():
    data0 = torch.randn(4,5)

    data1 =data0.reshape(5,4)

    print(data0.shape)
def reshape_view():
    data0 = torch.randn(4,5)

    data1 =data0.view(5,4)

    print(data0.shape)
    
def reshape_transpose():
    data0 = torch.randn(4,5) # stride = (5, 1) --> (2, 4, 3) --> (12, 3, 1)

    data1 =data0.T # 数据不会真正搬迁，但是stride 会变化。stride 对应做转置 ： （1，5）
    
    data2 = data1.contiguous() # 

    print(data0.shape)
    

if __name__ == '__main__':
    
    # reshape_demo()
    # reshape_view()
    reshape_transpose()
    print("run test_tensor.py successfully !!!")