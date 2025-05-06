import os
import hashlib
from collections import defaultdict

def get_file_hash(file_path):
    """计算文件内容的哈希值"""
    hash_md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def delete_duplicate_txt_files(directory='.'):
    """删除指定目录中内容重复的txt文件，保留一个"""
    # 获取所有txt文件
    txt_files = [f for f in os.listdir(directory) if f.endswith('.txt') and os.path.isfile(os.path.join(directory, f))]
    
    if not txt_files:
        print("当前目录没有找到txt文件")
        return
    
    # 按内容哈希值分组
    hash_to_files = defaultdict(list)
    for file_name in txt_files:
        file_path = os.path.join(directory, file_name)
        file_hash = get_file_hash(file_path)
        hash_to_files[file_hash].append(file_path)
    
    # 删除重复文件，保留每组的第一个
    deleted_count = 0
    for file_hash, file_paths in hash_to_files.items():
        if len(file_paths) > 1:
            print(f"发现内容相同的文件组:")
            for i, path in enumerate(file_paths):
                print(f"  {i+1}. {path}")
            
            # 保留第一个文件，删除其余文件
            for path in file_paths[1:]:
                print(f"删除: {path}")
                os.remove(path)
                deleted_count += 1
    
    print(f"\n操作完成！共删除了 {deleted_count} 个重复文件")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"正在检查目录: {current_dir}")
    delete_duplicate_txt_files(current_dir)
