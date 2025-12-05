
# data_loader.py   （2025 12月 终极简洁版，无 Retry 依赖，专注数据适配）
import akshare as ak
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import time
import random

def get_futures_data(symbol="RB0", days=500):
    """
    简洁版：直接用 AKShare + 列名适配 + 简单延时防限流
    """
    # 简单延时 + 随机 UA 伪装（不依赖 urllib3 重试）
    time.sleep(random.uniform(0.5, 1.5))  # 避免频繁请求
    
    try:
        df = ak.futures_main_sina(symbol=symbol)
        if df.empty:
            raise ValueError(f"无法获取 {symbol} 数据，请检查代码是否正确")
    except Exception as e:
        # 备用重试（纯 Python，无外部依赖）
        print(f"【警告】首次获取 {symbol} 失败（{str(e)}），重试中...")
        time.sleep(2)
        try:
            df = ak.futures_main_sina(symbol=symbol)
        except Exception as e2:
            raise ValueError(f"AKShare 接口重试也失败：{str(e2)}。建议：1. pip install akshare --upgrade；2. 检查网络/VPN；3. 试 AU0 等稳定品种。")

    # 列名适配（核心功能，保持不变）
    print(f"【调试信息】{symbol} 的原始列名：{list(df.columns)}")

    # 日期列
    date_col = None
    for col in ['日期', 'date', 'datetime', 'trade_date']:
        if col in df.columns:
            date_col = col
            break
    if date_col is None:
        raise ValueError(f"无法找到日期列。实际列名：{list(df.columns)}")

    df = df.rename(columns={date_col: 'date'})

    # 收盘价列
    close_col = None
    for col in ['收盘价', 'close', 'settle']:
        if col in df.columns:
            close_col = col
            break
    if close_col is None:
        raise ValueError(f"无法找到收盘价列。实际列名：{list(df.columns)}")

    df = df.rename(columns={close_col: 'close'})

    # 其他列
    col_mapping = {
        '开盘价': 'open', '最高价': 'high', '最低价': 'low',
        '成交量': 'volume', '持仓量': 'position'
    }
    df = df.rename(columns={k: v for k, v in col_mapping.items() if k in df.columns})

    # 数据清洗
    df = df.iloc[-days:].copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.normalize()
    df = df.dropna(subset=['date', 'close'])
    df = df.sort_values('date').reset_index(drop=True)

    if df.empty:
        raise ValueError(f"数据清洗后为空。请试试连续合约如 RB0")

    print(f"【调试信息】{symbol} 数据加载成功：{len(df)} 行，从 {df['date'].min().strftime('%Y-%m-%d')} 到 {df['date'].max().strftime('%Y-%m-%d')}")
    return df

# 在 data_loader.py 中修改 prepare_data 函数：

def prepare_data(df, seq_len=60, require_min_data=True, min_samples=10):
    """
    准备训练数据 - 修复版：处理数据不足的情况
    
    参数:
        df: 数据DataFrame
        seq_len: 序列长度
        require_min_data: 是否要求最小数据量，如果为False，会在数据不足时自动调整
        min_samples: 最小训练样本数
    """
    # 如果数据太少，自动调整
    if len(df) < seq_len + min_samples:
        if require_min_data:
            raise ValueError(f"数据不足 ({len(df)} < {seq_len+min_samples})，无法创建序列")
        else:
            # 自动调整seq_len以适应可用数据
            new_seq_len = max(10, min(seq_len, len(df) - min_samples))
            if new_seq_len < seq_len:
                print(f"⚠️  数据不足 ({len(df)} < {seq_len+min_samples})，自动调整序列长度: {seq_len} -> {new_seq_len}")
                seq_len = new_seq_len
            if seq_len < 10:
                # 即使调整后仍然不足，尝试使用所有数据
                print(f"数据严重不足 ({len(df)} 行)，使用所有数据")
                seq_len = max(5, len(df) // 2)
    
    scaler = MinMaxScaler()
    close_price = df['close'].values.reshape(-1, 1)
    scaled_data = scaler.fit_transform(close_price)

    X, y = [], []
    for i in range(seq_len, len(scaled_data)):
        X.append(scaled_data[i-seq_len:i, 0])
        y.append(scaled_data[i, 0])

    if len(X) == 0:
        # 如果数据太少，创建一个序列
        if len(scaled_data) >= seq_len:
            # 使用所有数据创建一个序列
            X.append(scaled_data[:seq_len, 0])
            y.append(scaled_data[-1, 0])
        else:
            # 填充数据
            padding_needed = seq_len - len(scaled_data)
            padded_data = np.concatenate([scaled_data[:, 0]] * (padding_needed // len(scaled_data) + 1))
            padded_data = padded_data[:seq_len]
            X.append(padded_data)
            y.append(scaled_data[-1, 0])
    
    X = np.array(X)
    y = np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # 分割训练集和验证集
    if len(X) > 20:
        train_X = X[:-10]  # 留出10个样本作为验证
        train_y = y[:-10]
    else:
        # 如果数据太少，使用全部数据
        train_X = X
        train_y = y
    
    print(f"数据准备完成: 序列长度={seq_len}, 总样本数={len(X)}, 训练样本数={len(train_X)}")
    
    return train_X, train_y, scaler, X, close_price