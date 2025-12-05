
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data_loader import prepare_data
import pandas as pd

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # 线性变换并分割成多头
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        energy = torch.matmul(Q, K.transpose(-2, -1)) / self.scale.to(x.device)
        attention = torch.softmax(energy, dim=-1)
        attention = self.dropout(attention)
        
        # 应用注意力权重
        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # 最终线性变换
        out = self.fc_out(out)
        return out, attention

class LSTMMAPModel(nn.Module):
    """LSTM-MAP: LSTM with Multi-head Attention Pooling"""
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, 
                 num_heads=4, dropout=0.2, output_size=1):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # 使用双向LSTM
        )
        
        # 注意力机制
        self.attention = MultiHeadAttention(hidden_size * 2, num_heads, dropout)  # 双向所以*2
        
        # 层归一化 - 修复：正确设置维度
        self.ln1 = nn.LayerNorm(hidden_size * 2)  # 输入维度是 hidden_size * 2
        self.ln2 = nn.LayerNorm(hidden_size * 3)  # 修复：应该是 hidden_size * 3
        
        # 注意力池化
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
        
        # 时间特征提取
        self.time_aware = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size * 2),  # 修复：增加维度
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, output_size)
        )
        
        # 残差连接 - 修复：确保维度匹配
        self.residual = nn.Linear(hidden_size * 2, hidden_size * 3)
        
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # LSTM编码
        lstm_out, (hidden, cell) = self.lstm(x)  # [batch, seq_len, hidden*2]
        lstm_out = self.dropout_layer(lstm_out)
        
        # 注意力机制
        attn_out, attention_weights = self.attention(lstm_out)  # [batch, seq_len, hidden*2]
        attn_out = self.ln1(attn_out + lstm_out)  # 残差连接
        
        # 注意力池化 - 获取每个时间步的重要性权重
        attention_scores = self.attention_pool(attn_out)  # [batch, seq_len, 1]
        attention_scores = F.softmax(attention_scores, dim=1)
        
        # 加权池化
        context_vector = torch.sum(attn_out * attention_scores, dim=1)  # [batch, hidden*2]
        
        # 时间特征提取
        time_features = self.time_aware(attn_out[:, -1, :])  # 取最后时间步 [batch, hidden]
        
        # 结合所有特征
        combined = torch.cat([context_vector, time_features], dim=1)  # [batch, hidden*3]
        
        # 残差连接
        residual = self.residual(lstm_out[:, -1, :])  # [batch, hidden*3]
        combined = self.ln2(combined + residual)
        
        # 最终预测
        output = self.fc(combined)
        
        return output, attention_scores.squeeze(-1)

# 在 lstm_model.py 的 train_and_predict 函数中修改数据准备部分：
# 在 lstm_model.py 的 train_and_predict 函数中修改训练部分：

def train_and_predict(symbol="RB2410", seq_len=60, epochs=50, 
                      hidden_size=64, num_layers=2, num_heads=4, 
                      learning_rate=0.001, dropout=0.2, algorithm="lstm_map",
                      data_days=400):  # 新增参数
    import os
    from data_loader import get_futures_data, prepare_data
    
    if algorithm == "rf_mip_lstm":
        from rf_mip_lstm import train_and_predict_rf_mip_lstm
        return train_and_predict_rf_mip_lstm(
            symbol=symbol,
            seq_len=seq_len,
            epochs=epochs
        )
    
    model_path = f"models/{symbol}_lstm_map.pth"
    best_model_path = model_path.replace('.pth', '_best.pth')
    
    # 使用传入的 data_days 参数
    df = get_futures_data(symbol, days=data_days)
    
    # 剩下的代码保持不变...
    
    # 检查数据是否足够
    if len(df) < seq_len + 30:
        print(f"⚠️  数据不足 ({len(df)} < {seq_len+30})，自动调整序列长度")
        seq_len = max(20, min(seq_len, len(df) - 30))
        print(f"调整后序列长度: {seq_len}")
    
    train_X, train_y, scaler, all_X, real_prices = prepare_data(df, seq_len, require_min_data=False)
    
    # 转换为PyTorch tensor - 修复：添加这行代码
    train_X_tensor = torch.FloatTensor(train_X)
    train_y_tensor = torch.FloatTensor(train_y).unsqueeze(1)  # 添加维度匹配输出
    
    # 创建LSTM-MAP模型
    model = LSTMMAPModel(
        input_size=1,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        output_size=1
    )
    
    # 损失函数和优化器
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    try:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
    except TypeError:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
    
    # 训练（如果已有模型直接加载）
    model_trained = False
    if os.path.exists(model_path):
        try:
            print(f"尝试加载已训练模型: {model_path}")
            
            # 方法1：直接加载整个模型（适用于参数完全匹配的情况）
            try:
                model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
                model.eval()
                model_trained = True
                print(f"✅ 成功加载已训练模型")
            except Exception as e1:
                print(f"严格模式加载失败: {e1}")
                
                # 方法2：尝试加载最佳模型
                if os.path.exists(best_model_path):
                    try:
                        model.load_state_dict(torch.load(best_model_path, map_location='cpu'), strict=False)
                        model.eval()
                        model_trained = True
                        print(f"✅ 成功加载最佳模型（宽松模式）")
                    except Exception as e2:
                        print(f"最佳模型加载失败: {e2}")
                        model_trained = False
                
        except Exception as e:
            print(f"模型加载失败: {e}")
            model_trained = False
    
    # 如果模型加载失败或不存在，重新训练
    if not model_trained:
        print(f"正在为 {symbol} 训练LSTM-MAP模型...")
        print(f"模型参数: hidden_size={hidden_size}, num_layers={num_layers}, num_heads={num_heads}")
        
        model.train()
        best_loss = float('inf')
        patience_counter = 0
        early_stop_patience = 15
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            # 修复：使用tensor而不是numpy数组
            output, attention_scores = model(train_X_tensor)
            loss = criterion(output, train_y_tensor)
            
            # 添加注意力正则化
            attention_reg = torch.mean(torch.sum(attention_scores ** 2, dim=1))
            total_loss = loss + 0.01 * attention_reg
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step(total_loss)
            
            if (epoch+1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}, "
                      f"Total Loss: {total_loss.item():.6f}, "
                      f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # 早停机制
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                
            if patience_counter >= early_stop_patience:
                print(f"早停于 Epoch {epoch+1}")
                if os.path.exists(best_model_path):
                    model.load_state_dict(torch.load(best_model_path, map_location='cpu'))
                break
        
        # 保存最终模型
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"LSTM-MAP模型已保存至 {model_path}")
    
    # 预测未来 30 天
    model.eval()
    future_pred = []
    attention_history = []
    
    with torch.no_grad():
        input_seq = all_X[-1:].copy()
        
        for step in range(30):
            # 修复：将numpy数组转换为tensor
            input_tensor = torch.FloatTensor(input_seq)
            pred, attention_scores = model(input_tensor)
            
            future_pred.append(pred.numpy()[0, 0])
            attention_history.append(attention_scores.numpy()[0])
            
            # 滑动窗口
            new_seq = np.append(input_seq[0, 1:, :], [[pred.numpy()[0, 0]]], axis=0)
            input_seq = new_seq.reshape(1, seq_len, 1)
    
    # 反标准化
    future_pred = scaler.inverse_transform(np.array(future_pred).reshape(-1, 1))
    future_pred = future_pred.flatten().tolist()
    
    # 历史真实价格 + 预测价格
    hist_dates = df['date'].dt.strftime('%Y-%m-%d').tolist()[-120:]
    hist_prices = real_prices[-120:].flatten().tolist()
    
    # 生成未来日期
    last_date = df['date'].iloc[-1]
    future_dates = [(last_date + pd.Timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(30)]
    
    # 计算模型置信度
    future_pred_np = np.array(future_pred)
    confidence = 1.0 / (1.0 + np.std(future_pred_np[:10]))
    
    return {
        "symbol": symbol,
        "hist_dates": hist_dates,
        "hist_prices": hist_prices,
        "future_dates": future_dates,
        "future_prices": future_pred,
        "model_type": "LSTM-MAP",
        "confidence": float(confidence),
        "attention_weights": attention_history[-1].tolist() if attention_history else [],
        "current_price": float(df['close'].iloc[-1]),
        "change": float(df['close'].iloc[-1] - df['close'].iloc[-2]),
        "change_pct": float((df['close'].iloc[-1] / df['close'].iloc[-2] - 1) * 100),
        "volume": int(df['volume'].iloc[-1]) if 'volume' in df.columns else 0,
        "open_interest": int(df.get('position', df.get('hold', pd.Series([0]*len(df)))).iloc[-1]),
        "open": float(df['open'].iloc[-1]) if 'open' in df.columns else 0,
        "high": float(df['high'].iloc[-1]) if 'high' in df.columns else 0,
        "low": float(df['low'].iloc[-1]) if 'low' in df.columns else 0,
        "prev_close": float(df['close'].iloc[-2]) if len(df) > 1 else 0,
        "timestamp": str(df['date'].iloc[-1]),
        "used_existing_model": model_trained
    }