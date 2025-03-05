'''
1. QKV 三个linear 要用上；
2. logits 没有归一化；
3. label 也不太标准；
4. 没用残差连接；
5. 没用layernorm;
6. relu 改为gelu;
7. decoder 的层数增加(kv cache 多层)；
8. dmodle 和 n_head 改变；
9. 数据集增加；
10. 问答类型的数据增加；
'''
import torch
import torch.nn as nn
import torch.optim as optim

class CausalMiniLlamaKV(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_head=2):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        
        # 嵌入层
        self.embed = nn.Embedding(vocab_size, d_model)
        
        # 自注意力投影层
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # 自注意力计算
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_head,
            batch_first=False
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.ReLU(),
            nn.Linear(d_model*4, d_model)
        )
        
        # 输出层
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        # 缓存因果掩码（动态生成）
        self.causal_mask = None

    def _generate_causal_mask(self, sz):
        return torch.triu(torch.ones(sz, sz) == 1, diagonal=1).bool()

    def forward(self, x, past_key_values=None):
        # 输入形状: [seq_len, batch_size]
        seq_len = x.size(0)
        x = self.embed(x)  # [seq_len, batch, d_model]
        
        # 计算Q/K/V投影
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # 处理KV缓存
        if past_key_values is not None:
            past_K, past_V = past_key_values
            K = torch.cat([past_K, K], dim=0)
            V = torch.cat([past_V, V], dim=0)
        
        # 生成因果掩码
        if past_key_values is None:
            # 训练阶段使用完整掩码
            if self.causal_mask is None or self.causal_mask.size(0) != seq_len:
                self.causal_mask = self._generate_causal_mask(seq_len).to(x.device)
            attn_mask = self.causal_mask[:seq_len, :seq_len]
        else:
            # 推理阶段生成动态掩码
            attn_mask = torch.zeros((seq_len, K.size(0)), 
                             dtype=torch.bool, device=x.device)
            attn_mask[:, -seq_len:] = True  # 屏蔽新加入的位置
        
        # 执行注意力计算
        attn_out, _ = self.self_attn(
            query=Q,
            key=K,
            value=V,
            attn_mask=attn_mask if seq_len > 1 else None,
            need_weights=False
        )
        
        ffn_out = self.ffn(attn_out)
        logits = self.lm_head(ffn_out)
        
        return logits, (K, V)

# 生成函数（使用KV Cache）
def generate_kv(prompt, max_len=50):
    model.eval()
    input_ids = tokenizer.encode(prompt)
    
    # 填充对齐
    if len(input_ids) < seq_length:
        pad_id = tokenizer.vocab['<PAD>']
        input_ids = [pad_id]*(seq_length - len(input_ids)) + input_ids
    else:
        input_ids = input_ids[-seq_length:]
    
    eos_id = tokenizer.vocab['<EOS>']
    past_key_values = None
    
    # 处理初始输入
    with torch.no_grad():
        # 初始前向传播获取缓存
        initial_input = torch.tensor(input_ids).unsqueeze(1)
        logits, past_key_values = model(initial_input)
        
        # 取最后一个token作为初始输入
        # next_id = input_ids[-1]
        next_id = torch.argmax(logits[-1, 0]).item()
        
        
        for _ in range(max_len):
            # 准备单步输入
            x = torch.tensor([[next_id]])
            
            # 前向传播
            logits, new_kv = model(x, past_key_values)
            
            # 更新缓存
            past_key_values = new_kv
            
            # 获取预测结果
            next_id = torch.argmax(logits[-1, 0]).item()
            input_ids.append(next_id)
            
            if next_id == eos_id:
                break
    
    return tokenizer.decode(input_ids).split('<EOS>')[0] + '<EOS>'

# 使用之前定义的分词器和训练流程（需稍作调整）
class CharTokenizer:
    def __init__(self, corpus):
        self.chars = ['<PAD>', '<EOS>'] + sorted(list(set(corpus)))
        self.vocab = {c:i for i,c in enumerate(self.chars)}
        self.ivocab = {i:c for i,c in enumerate(self.chars)}
    
    def encode(self, text):
        return [self.vocab.get(c, self.vocab['<PAD>']) for c in text]
    
    def decode(self, ids):
        return ''.join([self.ivocab[i] for i in ids if i != self.vocab['<PAD>']])
    
# 训练配置（保持不变）
corpus = ("中国的首都位于北京<EOS>北京是政治文化中心<EOS>首都有天安门<EOS>")
tokenizer = CharTokenizer(corpus)
vocab_size = len(tokenizer.chars)
seq_length = 5

# 初始化模型（注意使用新类）
model = CausalMiniLlamaKV(vocab_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 数据预处理（滑动窗口）
sentences = corpus.split('<EOS>')[:-1]
inputs, targets = [], []
for sent in sentences:
    sent += '<EOS>'
    for i in range(len(sent) - seq_length):
        inputs.append(sent[i:i+seq_length])
        targets.append(sent[i+1:i+1+seq_length])

# 训练循环（带因果注意力）
for epoch in range(100):
    total_loss = 0
    for seq_in, seq_out in zip(inputs, targets):
        x = torch.tensor(tokenizer.encode(seq_in)).unsqueeze(1)  # [seq_len, 1]
        y = torch.tensor(tokenizer.encode(seq_out))
        
        optimizer.zero_grad()
        logits, kv = model(x)
        loss = nn.CrossEntropyLoss()(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(inputs):.4f}")

# 测试生成
print(generate_kv("中国的首"))  # 使用KV Cache加速生成