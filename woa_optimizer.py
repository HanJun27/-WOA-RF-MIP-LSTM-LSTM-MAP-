import numpy as np
import torch
import torch.nn as nn
from lstm_model import LSTMMAPModel
from rf_mip_lstm import RFMIPLSTMModel, RFMIPLSTMPredictor
import pandas as pd
import json
import os
from sklearn.preprocessing import StandardScaler
from data_loader import get_futures_data, prepare_data
import warnings
warnings.filterwarnings('ignore')

class WOAOptimizer:
    """鲸鱼优化算法 (Whale Optimization Algorithm)"""
    def __init__(self, algorithm_type='lstm_map', population_size=15, max_iterations=30, 
                 data_days=200):
        self.algorithm_type = algorithm_type
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.data_days = data_days
        
        # 搜索空间定义
        if algorithm_type == 'lstm_map':
            self.search_space = {
                'hidden_size': (32, 128),      # 隐藏层大小
                'num_layers': (1, 3),          # LSTM层数
                'num_heads': (2, 8),           # 注意力头数
                'dropout': (0.1, 0.5),         # Dropout率
                'learning_rate': (1e-4, 1e-2), # 学习率
                'seq_len': (30, 90)            # 序列长度
            }
        else:  # rf_mip_lstm
            self.search_space = {
                'hidden_size': (32, 128),      # 隐藏层大小
                'num_layers': (1, 3),          # LSTM层数
                'dropout': (0.1, 0.5),         # Dropout率
                'learning_rate': (1e-4, 1e-2), # 学习率
                'seq_len': (30, 90)            # 序列长度
            }
        
        # 优化状态
        self.best_solution = None
        self.best_fitness = float('inf')
        self.convergence_history = []
        
    def initialize_population(self):
        """初始化种群"""
        population = []
        
        for _ in range(self.population_size):
            solution = {}
            for param, (min_val, max_val) in self.search_space.items():
                if param == 'hidden_size' and 'num_heads' in self.search_space:
                    # 确保hidden_size是num_heads的倍数
                    if 'num_heads' in self.search_space:
                        num_heads = int(np.random.randint(
                            self.search_space['num_heads'][0],
                            self.search_space['num_heads'][1] + 1
                        ))
                        # 选择一个num_heads的倍数
                        base = np.random.randint(
                            max(4, self.search_space['hidden_size'][0] // 8),
                            self.search_space['hidden_size'][1] // 8
                        )
                        solution['hidden_size'] = base * num_heads
                        solution['hidden_size'] = max(
                            self.search_space['hidden_size'][0],
                            min(solution['hidden_size'], self.search_space['hidden_size'][1])
                        )
                    else:
                        solution['hidden_size'] = int(np.random.randint(
                            self.search_space['hidden_size'][0],
                            self.search_space['hidden_size'][1] + 1
                        ))
                elif param == 'num_heads' and 'hidden_size' in solution:
                    # 确保num_heads是hidden_size的约数
                    hidden_size = solution['hidden_size']
                    possible_heads = []
                    for i in range(self.search_space['num_heads'][0], 
                                   self.search_space['num_heads'][1] + 1):
                        if hidden_size % i == 0:
                            possible_heads.append(i)
                    
                    if possible_heads:
                        solution['num_heads'] = np.random.choice(possible_heads)
                    else:
                        # 如果没有合适的约数，选择最近的
                        solution['num_heads'] = 4
                elif param in ['hidden_size', 'num_layers', 'num_heads', 'seq_len']:
                    # 整数参数
                    solution[param] = int(np.random.randint(min_val, max_val + 1))
                else:
                    # 浮点数参数
                    solution[param] = np.random.uniform(min_val, max_val)
            
            population.append(solution)
        
        return population
    
    def _adjust_lstm_map_params(self, solution):
        """调整LSTM-MAP参数，确保有效性"""
        adjusted = solution.copy()
        
        # 确保hidden_size能被num_heads整除
        if 'hidden_size' in adjusted and 'num_heads' in adjusted:
            if adjusted['hidden_size'] % adjusted['num_heads'] != 0:
                base = adjusted['hidden_size'] // adjusted['num_heads']
                adjusted['hidden_size'] = base * adjusted['num_heads']
                # 确保在范围内
                min_val, max_val = self.search_space['hidden_size']
                adjusted['hidden_size'] = max(min_val, min(adjusted['hidden_size'], max_val))
        
        # 确保num_layers在范围内
        if 'num_layers' in adjusted:
            min_val, max_val = self.search_space['num_layers']
            adjusted['num_layers'] = int(max(min_val, min(adjusted['num_layers'], max_val)))
        
        return adjusted
    
    def _fast_evaluate_lstm_map(self, solution, df, seq_len):
        """快速评估LSTM-MAP - 大大简化"""
        try:
            # 如果序列太长，限制最大长度
            if seq_len > 50:
                seq_len = 50
            
            # 简化评估：仅使用最后几行数据进行快速训练
            from data_loader import prepare_data
            
            try:
                train_X, train_y, _, _, _ = prepare_data(df, seq_len, require_min_data=False, min_samples=10)
            except:
                return 10.0
            
            if len(train_X) < 10:
                return 10.0
            
            # 只使用少量数据进行训练
            sample_size = min(20, len(train_X))
            indices = np.random.choice(len(train_X), sample_size, replace=False)
            train_X_sample = train_X[indices]
            train_y_sample = train_y[indices]
            
            # 转换为tensor
            X_tensor = torch.FloatTensor(train_X_sample)
            y_tensor = torch.FloatTensor(train_y_sample).unsqueeze(1)
            
            # 创建模型
            model = LSTMMAPModel(
                input_size=1,
                hidden_size=int(solution['hidden_size']),
                num_layers=int(solution['num_layers']),
                num_heads=int(solution['num_heads']),
                dropout=float(solution['dropout']),
                output_size=1
            )
            
            # 定义损失函数和优化器
            criterion = nn.HuberLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=float(solution['learning_rate']))
            
            # 极速训练（1个epoch）
            model.train()
            total_loss = 0
            num_batches = min(2, len(X_tensor))
            
            for i in range(num_batches):
                optimizer.zero_grad()
                output, _ = model(X_tensor[i:i+1])
                loss = criterion(output, y_tensor[i:i+1])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # 平均损失作为适应度
            fitness = total_loss / num_batches
            
            return fitness
            
        except Exception as e:
            print(f"快速评估失败: {e}")
            return 10.0
    
    def _fast_evaluate_rf_mip_lstm(self, solution, df, seq_len):
        """快速评估RF-MIP-LSTM"""
        try:
            # 创建预测器
            predictor = RFMIPLSTMPredictor(
                seq_len=int(seq_len),
                hidden_size=int(solution['hidden_size']),
                num_layers=int(solution['num_layers'])
            )
            
            # 快速训练
            try:
                # 只训练少量epochs
                predictor.train(df, epochs=5)
                
                # 使用部分数据进行验证
                val_size = min(50, len(df))
                if val_size > seq_len:
                    val_data = df.iloc[-val_size:].copy()
                    
                    # 进行短期预测验证
                    try:
                        predictions = predictor.predict(val_data, future_days=5)
                        
                        if len(predictions) > 0:
                            # 计算预测误差（简化）
                            if len(val_data) > 5:
                                actual_prices = val_data['close'].values[-5:]
                                pred_prices = np.array(predictions[:5])
                                if len(actual_prices) == len(pred_prices):
                                    mae = np.mean(np.abs(actual_prices - pred_prices))
                                    # 归一化误差
                                    normalized_error = mae / np.mean(actual_prices)
                                    return float(normalized_error)
                    except:
                        pass
            except:
                pass
            
            return 0.5  # 默认适应度值
            
        except Exception as e:
            print(f"RF-MIP-LSTM评估失败: {e}")
            return 1.0
    
    def evaluate_solution(self, solution, symbol="AU0", data_fraction=0.2):
        """评估解决方案的适应度 - 快速版"""
        try:
            if self.algorithm_type == 'lstm_map':
                solution = self._adjust_lstm_map_params(solution)
            
            # 使用指定的数据天数
            df = get_futures_data(symbol, days=self.data_days)
            
            seq_len = int(solution['seq_len'])
            
            # 如果数据太少，返回较大适应度值
            if len(df) < seq_len + 30:
                return 20.0
            
            # 使用更少的数据进行评估
            eval_size = min(150, len(df))
            df_eval = df.iloc[-eval_size:].copy()
            
            if self.algorithm_type == 'lstm_map':
                fitness = self._fast_evaluate_lstm_map(solution, df_eval, seq_len)
            else:
                fitness = self._fast_evaluate_rf_mip_lstm(solution, df_eval, seq_len)
            
            return fitness
            
        except Exception as e:
            print(f"评估解决方案失败: {e}")
            return 100.0
    
    def optimize(self, symbol="AU0", verbose=True):
        """执行WOA优化 - 这是关键方法！"""
        # 初始化种群
        population = self.initialize_population()
        population_size = len(population)
        
        # 评估初始种群
        fitness = []
        for i, solution in enumerate(population):
            if verbose and i % 5 == 0:
                print(f"评估初始解 {i+1}/{population_size}...")
            fit = self.evaluate_solution(solution, symbol)
            fitness.append(fit)
        
        # 找到最佳解
        best_idx = np.argmin(fitness)
        self.best_solution = population[best_idx]
        self.best_fitness = fitness[best_idx]
        
        if verbose:
            print(f"初始最佳适应度: {self.best_fitness:.6f}")
        
        # WOA主循环
        for iteration in range(self.max_iterations):
            if verbose and iteration % 5 == 0:
                print(f"迭代 {iteration+1}/{self.max_iterations}，最佳适应度: {self.best_fitness:.6f}")
            
            a = 2 - iteration * (2 / self.max_iterations)  # 线性递减
            
            for i in range(population_size):
                # 更新位置
                r1 = np.random.rand()
                r2 = np.random.rand()
                
                A = 2 * a * r1 - a
                C = 2 * r2
                
                # 包围猎物或气泡网攻击
                p = np.random.rand()
                
                if p < 0.5:
                    if abs(A) < 1:
                        # 包围猎物
                        for param in self.best_solution.keys():
                            if param in population[i]:
                                D = abs(C * self.best_solution[param] - population[i][param])
                                population[i][param] = self.best_solution[param] - A * D
                    else:
                        # 搜索猎物
                        random_idx = np.random.randint(0, population_size)
                        for param in self.best_solution.keys():
                            if param in population[i] and param in population[random_idx]:
                                D = abs(C * population[random_idx][param] - population[i][param])
                                population[i][param] = population[random_idx][param] - A * D
                else:
                    # 气泡网攻击（螺旋更新）
                    b = 1  # 螺旋形状常数
                    l = np.random.uniform(-1, 1)
                    
                    for param in self.best_solution.keys():
                        if param in population[i]:
                            D_prime = abs(self.best_solution[param] - population[i][param])
                            population[i][param] = D_prime * np.exp(b * l) * np.cos(2 * np.pi * l) + self.best_solution[param]
                
                # 确保参数在范围内
                for param, (min_val, max_val) in self.search_space.items():
                    if param in population[i]:
                        if param in ['hidden_size', 'num_layers', 'num_heads', 'seq_len']:
                            population[i][param] = int(max(min_val, min(population[i][param], max_val)))
                        else:
                            population[i][param] = max(min_val, min(population[i][param], max_val))
                
                # 评估新解
                new_fitness = self.evaluate_solution(population[i], symbol)
                
                # 更新个体最佳
                if new_fitness < fitness[i]:
                    fitness[i] = new_fitness
                    
                    # 更新全局最佳
                    if new_fitness < self.best_fitness:
                        self.best_solution = population[i].copy()
                        self.best_fitness = new_fitness
            
            self.convergence_history.append(self.best_fitness)
        
        # 最终调整参数
        if self.algorithm_type == 'lstm_map':
            self.best_solution = self._adjust_lstm_map_params(self.best_solution)
        
        if verbose:
            print(f"优化完成！最终适应度: {self.best_fitness:.6f}")
        
        return self.best_solution, self.best_fitness
    
    def format_solution(self, solution):
        """格式化解决方案为可读格式"""
        formatted = {}
        for param, value in solution.items():
            if param in ['hidden_size', 'num_layers', 'num_heads', 'seq_len']:
                formatted[param] = int(value)
            else:
                formatted[param] = float(f"{value:.6f}")
        return formatted
    
    def _get_default_solution(self):
        """获取默认解决方案"""
        if self.algorithm_type == 'lstm_map':
            return {
                'hidden_size': 64,
                'num_layers': 2,
                'num_heads': 4,
                'dropout': 0.2,
                'learning_rate': 0.001,
                'seq_len': 60
            }
        else:  # rf_mip_lstm
            return {
                'hidden_size': 64,
                'num_layers': 2,
                'dropout': 0.2,
                'learning_rate': 0.001,
                'seq_len': 60
            }
    
    def save_best_params(self):
        """保存最佳参数到文件"""
        os.makedirs("config", exist_ok=True)
        
        algorithm_str = self.algorithm_type.replace('_', '-')
        config_file = f"config/{algorithm_str}_best_params.json"
        
        config_data = {
            'algorithm': self.algorithm_type,
            'best_fitness': float(self.best_fitness),
            'best_params': self.format_solution(self.best_solution),
            'convergence_history': [float(f) for f in self.convergence_history],
            'optimization_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        print(f"最佳参数已保存至 {config_file}")

# 优化函数（保持原有函数不变）
def optimize_lstm_map_with_woa(symbol="AU0", population_size=12, max_iterations=25, data_days=200):
    """使用WOA优化LSTM-MAP的超参数 - 快速版"""
    print("=" * 60)
    print(f"开始使用WOA优化LSTM-MAP ({symbol})")
    print(f"使用数据天数: {data_days}, 种群大小: {population_size}, 迭代次数: {max_iterations}")
    print("=" * 60)
    
    optimizer = WOAOptimizer(
        algorithm_type='lstm_map',
        population_size=population_size,
        max_iterations=max_iterations,
        data_days=data_days
    )
    
    best_params, best_fitness = optimizer.optimize(symbol=symbol, verbose=True)
    
    print("\n" + "=" * 60)
    print("优化完成!")
    if np.isnan(best_fitness):
        print("⚠️  优化过程中出现NaN，使用默认参数")
        best_params = optimizer._get_default_solution()
        best_fitness = 0.1
    else:
        print(f"最佳适应度: {best_fitness:.6f}")
    
    print("最佳参数:")
    for param, value in optimizer.format_solution(best_params).items():
        print(f"  {param}: {value}")
    
    # 验证参数有效性
    if 'hidden_size' in best_params and 'num_heads' in best_params:
        if best_params['hidden_size'] % best_params['num_heads'] != 0:
            print(f"⚠️  警告: hidden_size({best_params['hidden_size']}) "
                  f"不能被num_heads({best_params['num_heads']})整除")
            adjusted = best_params['hidden_size'] // best_params['num_heads']
            best_params['hidden_size'] = adjusted * best_params['num_heads']
            print(f"自动调整为: hidden_size={best_params['hidden_size']}")
    
    # 保存结果
    optimizer.save_best_params()
    
    return best_params

def optimize_rf_mip_lstm_with_woa(symbol="AU0", population_size=12, max_iterations=25):
    """使用WOA优化RF-MIP-LSTM的超参数"""
    print("=" * 60)
    print(f"开始使用WOA优化RF-MIP-LSTM ({symbol})")
    print("=" * 60)
    
    optimizer = WOAOptimizer(
        algorithm_type='rf_mip_lstm',
        population_size=population_size,
        max_iterations=max_iterations
    )
    
    best_params, best_fitness = optimizer.optimize(symbol=symbol, verbose=True)
    
    print("\n" + "=" * 60)
    print("优化完成!")
    if np.isnan(best_fitness):
        print("⚠️  优化过程中出现NaN，使用默认参数")
        best_params = optimizer._get_default_solution()
        best_fitness = 0.1
    else:
        print(f"最佳适应度: {best_fitness:.6f}")
    
    print("最佳参数:")
    for param, value in optimizer.format_solution(best_params).items():
        print(f"  {param}: {value}")
    
    # 保存结果
    optimizer.save_best_params()
    
    return best_params

def get_optimized_params(algorithm_type='lstm_map', symbol="AU0", force_optimize=False):
    """
    获取优化后的参数（如果存在则加载，否则运行优化）
    """
    algorithm_str = algorithm_type.replace('_', '-')
    config_file = f"config/{algorithm_str}_best_params.json"
    
    # 检查是否已有优化参数
    if not force_optimize and os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"加载已优化的参数: {config_file}")
            return data['best_params']
        except:
            print(f"加载参数失败，将重新优化")
    
    # 运行优化（使用较小的参数加快速度）
    if algorithm_type == 'lstm_map':
        return optimize_lstm_map_with_woa(symbol, population_size=8, max_iterations=15)
    else:
        return optimize_rf_mip_lstm_with_woa(symbol, population_size=8, max_iterations=15)

# 测试函数
def test_woa_optimization():
    """测试WOA优化功能"""
    print("测试WOA优化功能...")
    
    # 测试LSTM-MAP优化
    print("\n1. 测试LSTM-MAP优化")
    optimizer = WOAOptimizer(algorithm_type='lstm_map', population_size=5, max_iterations=3)
    
    # 测试参数有效性调整
    test_solution = {
        'hidden_size': 36,
        'num_layers': 2,
        'num_heads': 4,
        'dropout': 0.2,
        'learning_rate': 0.001,
        'seq_len': 60
    }
    
    print(f"原始参数: {test_solution}")
    adjusted = optimizer._adjust_lstm_map_params(test_solution)
    print(f"调整后参数: {adjusted}")
    
    # 测试RF-MIP-LSTM优化
    print("\n2. 测试RF-MIP-LSTM优化")
    optimizer2 = WOAOptimizer(algorithm_type='rf_mip_lstm', population_size=5, max_iterations=3)
    
    print("\n3. 测试参数生成")
    solutions = optimizer.initialize_population()
    for i, sol in enumerate(solutions[:3]):
        print(f"解 {i+1}: hidden_size={sol['hidden_size']}, num_heads={sol['num_heads']}, "
              f"整除={sol['hidden_size'] % sol['num_heads'] == 0}")

if __name__ == "__main__":
    # 示例用法
    print("WOA优化器测试")
    test_woa_optimization()
    
    print("\n1. 优化LSTM-MAP")
    try:
        lstm_params = optimize_lstm_map_with_woa("AU0", population_size=6, max_iterations=10)
        print(f"优化结果: {lstm_params}")
    except Exception as e:
        print(f"优化失败: {e}")
    
    print("\n2. 获取优化参数（智能加载）")
    try:
        loaded_params = get_optimized_params('lstm_map', 'AU0')
        print(f"加载的参数: {loaded_params}")
    except Exception as e:
        print(f"加载失败: {e}")