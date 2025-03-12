import torch

torch.no_grad()

torch.optim
def grad_accumulate():
    # torch.seed()
    x = torch.ones(5)  # input tensor
    label = torch.zeros(3)  # expected output
    w = torch.randn(5, 3, requires_grad=True) # requires_grad
    b = torch.randn(3, requires_grad=True)    
    output = torch.matmul(x, w)+b # 全连接层 
    
    loss = torch.nn.functional.binary_cross_entropy_with_logits(output, label)
    loss.backward(retain_graph=True) # 反向传播：求梯度
    print(f"Grad for w first time = {w.grad}")
    # print(f"Gradient function for z = {output.grad_fn}")
    # print(f"Gradient function for loss = {loss.grad_fn}")
    # w.grad.zero_() # 清空梯度，直接置0
    # w.grad = None    # 置None，原tensor里的显存就释放掉了
    # with torch.no_grad():
    w.copy_(w - 0.01 * w.grad)
    
    # loss.backward(retain_graph=True) # 新算出来的结果，不是替换原来的值，而是累加到原来的值上
    print(f"Grad for w first time = {w.grad}")
    
    
def inplace_demo():
    data1 = torch.randn(3, 4)
    data1.requires_grad = True
    
    data2 = data1 + 2
    
    data2.mul_(2) # 直接+2
    loss = data2.var() # 
    
    loss.backward()
    
    
def inplace_demo_v2():
    # y = torch.randn(5, 5, requires_grad=True)
    
    with torch.no_grad():
        data1 = torch.randn(3, 4)
        data1.requires_grad = True
        
        data1.mul_(2)
        
        data1.backward(torch.randn_like(data1))

        # loss = data1.var() # 
        
        # loss.backward()

def autograd_demo_v1():
    torch.manual_seed(0) # 
    x = torch.ones(5, requires_grad=True) # input
    w = torch.randn(5, 5, requires_grad=True) # weight
    b = torch.randn_like(x)
    label = torch.Tensor([0, 0, 1, 0, 0])

    for i in range(100):
        # w.requires_grad=True # True 
        # if w.grad is not None:
        #   w.grad.zero_()
          
        z = torch.matmul(w, x) + b # linear layer    
        output = torch.sigmoid(z)
        # output.register_hook(hook)        
        output.retain_grad() # tensor([-0.0405, -0.0722, -0.1572,  0.3101, -0.0403]
        loss = (output-label).var() # l2 loss
        loss.backward()
        # print(w.grad)
        print("loss: ", loss)
        # w.sub_(0.05 * w.grad)
        # w = w - 0.8 * w.grad # 改了w 的属性了
        with torch.no_grad():
            w.sub_(0.05 * w.grad)
            
        w.grad =None
        
        # w.data.sub_(w.grad)
        # w.grad = None
        
        # print("w")
        # print("w")
        # w.retain_grad()
        # with torch.no_grad():
        #     w = w - 0.05 * w.grad
        
grad_list = []
def hook_func(grad):
    grad_list.append(grad)
    return grad + 5
    
    
# torch.Tensor
def hook_demo():
    # return 0.001*grad
    c = 5
    a = torch.Tensor([1, 2, 3])
    a.requires_grad = True
    a.register_hook(hook_func)
    b = a.mul(c)
    b.var().backward()
    import ipdb; ipdb.set_trace()
    print(f"==========")

class Exp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i.exp()
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        return grad_output * result

if __name__ == "__main__":
    # grad_accumulate()
    # inplace_demo()
    # inplace_demo_v2()
    # autograd_demo_v1()
    hook_demo()