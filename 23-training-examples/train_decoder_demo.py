import torch
import torch.nn as nn
import torch.optim as optim

class CausalMiniLlama(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_head=2):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        
        # 嵌入层
        self.embed = nn.Embedding(vocab_size, d_model)
        
        # 因果自注意力
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_head,
            batch_first=False  # 输入格式为 (seq_len, batch, features)
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.ReLU(),
            nn.Linear(d_model*4, d_model)
        )
        
        # 输出层
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        self.Q = nn.Linear(d_model, d_model)
        
        # 缓存因果掩码（动态生成）
        self.causal_mask = None

    def _generate_causal_mask(self, sz):
        """生成下三角布尔掩码 (False表示允许注意力)"""
        return torch.triu(torch.ones(sz, sz) == 1, diagonal=1).bool()

    def forward(self, x):
        # 输入形状: [seq_len, batch_size]
        seq_len = x.size(0)
        x = self.embed(x)  # [seq_len, batch, d_model]
        
        # 生成因果掩码
        if self.causal_mask is None or self.causal_mask.size(0) != seq_len:
            self.causal_mask = self._generate_causal_mask(seq_len).to(x.device)
        
        # 执行因果注意力
        attn_out, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=self.causal_mask[:seq_len, :seq_len]
        )
        
        ffn_out = self.ffn(attn_out)
        return self.lm_head(ffn_out)  # [seq_len, batch, vocab_size]

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

# 训练配置
corpus = ("中国的首都位于北京<EOS>北京是政治文化中心<EOS>首都有天安门<EOS>")
tokenizer = CharTokenizer(corpus)
vocab_size = len(tokenizer.chars)
seq_length = 5  # 输入序列长度

# 数据预处理（滑动窗口）
sentences = corpus.split('<EOS>')[:-1]
inputs, targets = [], []
for sent in sentences:
    sent += '<EOS>'
    for i in range(len(sent) - seq_length):
        inputs.append(sent[i:i+seq_length])
        targets.append(sent[i+1:i+1+seq_length])

# 初始化因果模型
model = CausalMiniLlama(vocab_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环（带因果注意力）
for epoch in range(100):
    total_loss = 0
    for seq_in, seq_out in zip(inputs, targets):
        x = torch.tensor(tokenizer.encode(seq_in)).unsqueeze(1)  # [seq_len, 1]
        y = torch.tensor(tokenizer.encode(seq_out))
        
        optimizer.zero_grad()
        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(inputs):.4f}")

# 生成函数（保持因果性）
def generate(prompt, max_len=50):
    model.eval()
    input_ids = tokenizer.encode(prompt)
    
    # 填充对齐
    if len(input_ids) < seq_length:
        pad_id = tokenizer.vocab['<PAD>']
        input_ids = [pad_id]*(seq_length - len(input_ids)) + input_ids
    else:
        input_ids = input_ids[-seq_length:]
    
    eos_id = tokenizer.vocab['<EOS>']
    
    with torch.no_grad():
        for _ in range(max_len):
            x = torch.tensor(input_ids[-seq_length:]).unsqueeze(1)
            logits = model(x)  # [seq_len, 1, vocab]
            
            # 只取最后一个位置的预测
            next_id = torch.argmax(logits[-1, 0]).item()
            input_ids.append(next_id) # 追加到input里
            
            if next_id == eos_id:
                break
    
    return tokenizer.decode(input_ids).split('<EOS>')[0] + '<EOS>'

# 测试生成
print(generate("中国的首"))  # 输出示例：中国的首都位于北京<EOS>