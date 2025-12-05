
"""
数据检查和修复工具
用于检查数据是否足够并自动调整参数
"""
import pandas as pd
import numpy as np
from data_loader import get_futures_data, prepare_data

def check_data_sufficiency(symbol="AU0", target_seq_len=60, min_samples=30):
    """
    检查数据是否足够进行训练
    
    参数:
        symbol: 品种代码
        target_seq_len: 目标序列长度
        min_samples: 所需最小样本数
    
    返回:
        dict: 检查结果
    """
    print(f"检查 {symbol} 数据充足性...")
    
    try:
        # 获取数据
        df = get_futures_data(symbol, days=500)
        print(f"获取到 {len(df)} 行数据")
        
        # 检查基本要求
        total_len = len(df)
        required_len = target_seq_len + min_samples
        
        if total_len < required_len:
            print(f"⚠️  数据不足: {total_len} < {required_len}")
            
            # 计算可用的最大seq_len
            max_seq_len = max(10, total_len - min_samples)
            if max_seq_len < 10:
                return {
                    'sufficient': False,
                    'total_data': total_len,
                    'required': required_len,
                    'max_seq_len': 0,
                    'message': '数据严重不足，无法训练'
                }
            
            recommended_seq_len = min(target_seq_len, max_seq_len)
            
            return {
                'sufficient': False,
                'total_data': total_len,
                'required': required_len,
                'max_seq_len': max_seq_len,
                'recommended_seq_len': recommended_seq_len,
                'message': f'建议调整序列长度: {target_seq_len} -> {recommended_seq_len}'
            }
        else:
            # 测试数据准备
            try:
                train_X, train_y, _, _, _ = prepare_data(df, target_seq_len)
                trainable_samples = len(train_X)
                
                return {
                    'sufficient': True,
                    'total_data': total_len,
                    'required': required_len,
                    'trainable_samples': trainable_samples,
                    'message': f'数据充足，可训练样本: {trainable_samples}'
                }
            except Exception as e:
                return {
                    'sufficient': False,
                    'total_data': total_len,
                    'required': required_len,
                    'message': f'数据准备失败: {str(e)}'
                }
                
    except Exception as e:
        return {
            'sufficient': False,
            'message': f'获取数据失败: {str(e)}'
        }

def auto_adjust_parameters(symbol="AU0", original_params=None):
    """
    根据数据情况自动调整参数
    
    参数:
        symbol: 品种代码
        original_params: 原始参数
    
    返回:
        dict: 调整后的参数
    """
    if original_params is None:
        original_params = {
            'hidden_size': 64,
            'num_layers': 2,
            'num_heads': 4,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'seq_len': 60
        }
    
    print(f"为 {symbol} 自动调整参数...")
    
    # 检查数据
    check_result = check_data_sufficiency(symbol, original_params['seq_len'])
    
    adjusted_params = original_params.copy()
    
    if not check_result['sufficient']:
        if 'recommended_seq_len' in check_result:
            adjusted_params['seq_len'] = check_result['recommended_seq_len']
            print(f"调整序列长度: {original_params['seq_len']} -> {adjusted_params['seq_len']}")
        
        # 根据数据量调整模型复杂度
        if check_result.get('total_data', 0) < 100:
            # 数据很少，使用简单模型
            adjusted_params['hidden_size'] = 32
            adjusted_params['num_layers'] = 1
            adjusted_params['num_heads'] = 2
            print(f"数据较少，简化模型: hidden_size=32, num_layers=1, num_heads=2")
    
    # 确保hidden_size能被num_heads整除
    if adjusted_params['hidden_size'] % adjusted_params['num_heads'] != 0:
        base = adjusted_params['hidden_size'] // adjusted_params['num_heads']
        adjusted_params['hidden_size'] = base * adjusted_params['num_heads']
        print(f"调整hidden_size为num_heads的倍数: {adjusted_params['hidden_size']}")
    
    return adjusted_params

def batch_check_symbols(symbols=None):
    """
    批量检查多个品种的数据情况
    
    参数:
        symbols: 品种列表
    
    返回:
        dict: 各品种检查结果
    """
    if symbols is None:
        symbols = ["AU0", "AG0", "CU0", "RB0", "SC0", "M0"]
    
    results = {}
    
    for symbol in symbols:
        print(f"\n检查 {symbol}...")
        result = check_data_sufficiency(symbol)
        results[symbol] = result
        
        if result['sufficient']:
            print(f"✅ {symbol}: {result['message']}")
        else:
            print(f"❌ {symbol}: {result['message']}")
    
    return results

def create_synthetic_data_if_needed(symbol="AU0", min_days=200):
    """
    如果真实数据不足，创建合成数据用于测试
    
    参数:
        symbol: 品种代码
        min_days: 最小所需天数
    
    返回:
        DataFrame: 真实数据或合成数据
    """
    try:
        df = get_futures_data(symbol, days=min_days*2)
        
        if len(df) >= min_days:
            print(f"✅ {symbol}: 有 {len(df)} 天真实数据")
            return df
        else:
            print(f"⚠️  {symbol}: 只有 {len(df)} 天数据，创建合成数据补充...")
            
            # 基于现有数据创建合成数据
            if len(df) > 0:
                # 有部分真实数据，基于此扩展
                last_date = df['date'].iloc[-1]
                last_price = df['close'].iloc[-1]
                
                # 创建合成数据
                synth_days = min_days - len(df)
                synth_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=synth_days)
                
                # 模拟价格波动
                synth_prices = []
                current_price = last_price
                for _ in range(synth_days):
                    # 随机波动
                    change = np.random.normal(0, last_price * 0.02)
                    current_price = max(0.1, current_price + change)
                    synth_prices.append(current_price)
                
                synth_df = pd.DataFrame({
                    'date': synth_dates,
                    'open': [p * 0.99 for p in synth_prices],
                    'high': [p * 1.02 for p in synth_prices],
                    'low': [p * 0.98 for p in synth_prices],
                    'close': synth_prices,
                    'volume': np.random.randint(10000, 100000, synth_days)
                })
                
                # 合并数据
                combined_df = pd.concat([df, synth_df], ignore_index=True)
                print(f"创建了 {synth_days} 天合成数据，总数据: {len(combined_df)} 天")
                
                return combined_df
            else:
                # 完全没有数据，创建完全合成的数据
                print(f"❌ {symbol}: 无真实数据，创建完全合成数据...")
                
                dates = pd.date_range(end=pd.Timestamp.now(), periods=min_days, freq='D')
                base_price = 1000 if 'AU' in symbol else 5000  # 根据品种设定基准价
                
                prices = base_price + np.random.randn(min_days).cumsum() * base_price * 0.02
                prices = np.abs(prices)  # 确保价格为正
                
                synth_df = pd.DataFrame({
                    'date': dates,
                    'open': prices * 0.99,
                    'high': prices * 1.02,
                    'low': prices * 0.98,
                    'close': prices,
                    'volume': np.random.randint(10000, 100000, min_days)
                })
                
                print(f"创建了 {min_days} 天合成数据")
                return synth_df
                
    except Exception as e:
        print(f"获取数据失败，创建合成数据: {e}")
        
        # 创建完全合成的数据
        dates = pd.date_range(end=pd.Timestamp.now(), periods=min_days, freq='D')
        prices = 1000 + np.random.randn(min_days).cumsum() * 50
        
        synth_df = pd.DataFrame({
            'date': dates,
            'open': prices * 0.99,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(10000, 100000, min_days)
        })
        
        print(f"创建了 {min_days} 天合成数据")
        return synth_df

if __name__ == "__main__":
    print("数据检查工具")
    print("=" * 50)
    
    # 检查主要品种
    results = batch_check_symbols()
    
    print("\n" + "=" * 50)
    print("参数自动调整示例:")
    
    # 测试参数调整
    test_params = {
        'hidden_size': 64,
        'num_layers': 2,
        'num_heads': 4,
        'dropout': 0.2,
        'learning_rate': 0.001,
        'seq_len': 60
    }
    
    adjusted = auto_adjust_parameters("AU0", test_params)
    print(f"调整后参数: {adjusted}")
    
    print("\n" + "=" * 50)
    print("数据补充示例:")
    
    # 测试数据补充
    df = create_synthetic_data_if_needed("AU0", min_days=200)
    print(f"最终数据量: {len(df)} 天")
