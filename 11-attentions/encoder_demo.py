import torch
from transformers import BertModel, BertConfig
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s - %(funcName)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SingleLayerBertEncoder(torch.nn.Module):
    def __init__(self, config):
        super(SingleLayerBertEncoder, self).__init__()
        self.embeddings = BertModel.from_pretrained('bert-base-uncased', config=config).embeddings
        self.encoder_layer = BertModel.from_pretrained('bert-base-uncased', config=config).encoder.layer[0]

    def forward(self, input_ids, attention_mask=None):
        # 获取嵌入输出
        embedding_output = self.embeddings(input_ids)
        # 使用单层编码器进行处理
        encoder_outputs = self.encoder_layer(hidden_states=embedding_output, 
                                             attention_mask=attention_mask)
        return encoder_outputs[0]  # 返回最后一层隐藏状态


def export_encoder_onnx():
    # 加载预训练的BERT配置
    config = BertConfig.from_pretrained('bert-base-uncased')
    model = SingleLayerBertEncoder(config)

    # 设置模型为评估模式
    model.eval()

    # 准备示例输入数据
    input_ids = torch.tensor([[101, 2023, 2003, 1037, 7354, 102]])  # 示例输入ID
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1]])  # 示例注意力掩码

    # 导出模型到ONNX
    torch.onnx.export(model,
                    args=(input_ids, attention_mask),
                    f="single_layer_bert_encoder.onnx",
                    input_names=['input_ids', 'attention_mask'],
                    output_names=['output'],
                    opset_version=11,
                    do_constant_folding=True,
                    dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence'},
                                    'attention_mask': {0: 'batch_size', 1: 'sequence'},
                                    'output': {0: 'batch_size', 1: 'sequence'}})

    logger.info("单层BERT编码器已成功导出为ONNX格式")
    
def run_encoder_onnx():
    from transformers import BertTokenizer
    import numpy as np
    import onnxruntime as ort

    # 加载分词器
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # 准备输入文本
    text = "Here is a sample sentence for the encoder."
    inputs = tokenizer(text, return_tensors='pt')

    # 将PyTorch张量转换为NumPy数组
    input_ids = inputs['input_ids'].numpy()
    attention_mask = inputs['attention_mask'].numpy()

    # 创建ONNX运行时会话
    ort_session = ort.InferenceSession("single_layer_bert_encoder.onnx")

    # 运行模型
    outputs = ort_session.run(
        None,  # 计算图中的输出节点名称；None表示返回所有输出
        {"input_ids": input_ids, "attention_mask": attention_mask},
    )

    # 输出是一个列表，其中包含了模型的所有输出
    output = outputs[0]

    logger.info("Model output:", output)
    
def onnx_shape_inference():
    import onnx
    from onnx import shape_inference

    # 加载原始模型
    model_path = "single_layer_bert_encoder.onnx"
    model = onnx.load(model_path)

    # 对模型进行形状推理
    inferred_model = shape_inference.infer_shapes(model)

    # 保存带有形状信息的模型（可选）
    onnx.save(inferred_model, "single_layer_bert_encoder_with_shapes.onnx")

    # 打印模型的计算图及形状信息
    logger.info(onnx.helper.printable_graph(inferred_model.graph))
    
if __name__ == '__main__':
    export_encoder_onnx() # 导出模型
    # run_encoder_onnx()
    # onnx_shape_inference()