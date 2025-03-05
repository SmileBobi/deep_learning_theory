'''
Llama 2 是一个 纯 Decoder 架构 的模型，没有 Encoder。

每个 Decoder 层包含 Masked Self-Attention 和 Feed-Forward Network。

使用因果掩码（Causal Mask）确保模型在生成时只能看到当前及之前的位置。
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class Llama2DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(Llama2DecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, tgt_mask=None):
        # Masked Self-Attention
        attn_output, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = tgt + self.dropout(attn_output)
        tgt = self.norm1(tgt)

        # Feed-Forward Network
        ff_output = self.linear2(self.dropout(F.gelu(self.linear1(tgt))))
        tgt = tgt + self.dropout(ff_output)
        tgt = self.norm2(tgt)

        return tgt

class Llama2Decoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super(Llama2Decoder, self).__init__()
        self.layers = nn.ModuleList([
            Llama2DecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, tgt, tgt_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, tgt_mask)
        return tgt
    
def decoder_run():
    # 定义模型参数
    d_model = 512
    nhead = 8
    num_layers = 1
    dim_feedforward = 2048
    dropout = 0.1

    # 实例化模型
    model = Llama2Decoder(d_model, nhead, num_layers, dim_feedforward, dropout)

    # 创建示例输入
    tgt = torch.rand(10, 32, d_model)  # (sequence_length, batch_size, d_model)
    tgt_mask = torch.triu(torch.ones(10, 10) * float('-inf'), diagonal=1)  # 因果掩码

    # 前向传播
    output = model(tgt, tgt_mask)
    print(output.shape)  # 输出形状: (10, 32, 512)
    
def onnx_export():
    # 定义模型参数
    d_model = 512
    nhead = 8
    num_layers = 6
    dim_feedforward = 2048
    dropout = 0.1

    # 实例化模型
    model = Llama2Decoder(d_model, nhead, num_layers, dim_feedforward, dropout)

    # 设置模型为评估模式
    model.eval()

    # 创建示例输入（固定形状）
    sequence_length = 10
    batch_size = 32
    tgt = torch.rand(sequence_length, batch_size, d_model)  # 固定形状 (10, 32, 512)
    tgt_mask = torch.triu(torch.ones(sequence_length, sequence_length) * float('-inf'), diagonal=1)  # 固定形状 (10, 10)

    # 导出模型为 ONNX 格式（静态形状）
    torch.onnx.export(
        model,  # 模型
        (tgt, tgt_mask),  # 模型输入（元组形式）
        "llama2_decoder_static.onnx",  # 导出的 ONNX 文件名
        input_names=["tgt", "tgt_mask"],  # 输入名称
        output_names=["output"],  # 输出名称
        opset_version=13,  # ONNX opset 版本
        verbose=True  # 打印导出日志
    )

    print("模型已成功导出为 llama2_decoder_static.onnx")
    
def onnx_shape_infer_and_simplify():
    import onnx
    from onnx import shape_inference
    from onnxsim import simplify

    # 加载导出的 ONNX 模型
    onnx_model = onnx.load("llama2_decoder_static.onnx")

    # 进行形状推理
    onnx_model = shape_inference.infer_shapes(onnx_model)
    
    # 简化模型
    simplified_model, check = simplify(onnx_model)
    # 检查简化是否成功
    if check:
        print("模型简化成功！")
    else:
        print("模型简化失败！")

    # 保存简化后的模型
    onnx.save(simplified_model, "llama2_decoder_static_shaped_simplified.onnx")

    
if __name__ == '__main__':
    # decoder_run()
    # onnx_export()
    onnx_shape_infer_and_simplify()
    

    
    
