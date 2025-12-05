# cleanup_models.py
import os
import shutil
import glob

def cleanup_old_models():
    """清理旧的模型文件，避免架构不匹配问题"""
    model_dir = "models"
    
    if os.path.exists(model_dir):
        # 备份旧模型
        backup_dir = "models_backup"
        os.makedirs(backup_dir, exist_ok=True)
        
        # 备份所有模型文件
        for model_file in glob.glob(os.path.join(model_dir, "*.pth")):
            try:
                shutil.move(model_file, backup_dir)
                print(f"已备份: {model_file}")
            except Exception as e:
                print(f"备份失败 {model_file}: {e}")
        
        # 清理配置文件
        config_dir = "config"
        if os.path.exists(config_dir):
            config_backup = "config_backup"
            os.makedirs(config_backup, exist_ok=True)
            
            for config_file in glob.glob(os.path.join(config_dir, "*.json")):
                try:
                    shutil.move(config_file, config_backup)
                    print(f"已备份: {config_file}")
                except Exception as e:
                    print(f"备份失败 {config_file}: {e}")
    
    print("清理完成！所有旧模型已备份到 models_backup 和 config_backup 目录")

if __name__ == "__main__":
    cleanup_old_models()