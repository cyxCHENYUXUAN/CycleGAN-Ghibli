import os
import shutil
import random

def split_dataset(source_dir, train_dir, test_dir, num_test_files=100):
    """
    从源文件夹中划分文件到训练集和测试集文件夹。
    在CycleGAN中，通常原始文件夹就是训练集文件夹，我们从中抽取图片到测试集。

    Args:
        source_dir (str): 包含所有图片的原始文件夹路径。
        train_dir (str): 训练集文件夹路径 (通常和source_dir相同)。
        test_dir (str): 测试集文件夹路径。
        num_test_files (int): 要抽取作为测试集的图片数量。
    """
    # 确保测试文件夹存在，如果不存在则创建
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        print(f"创建文件夹: {test_dir}")

    # 获取所有图片文件名
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

    # 检查是否有足够的文件用于测试集
    if len(files) < num_test_files:
        raise ValueError(f"文件夹 {source_dir} 中只有 {len(files)} 张图片，不足以抽取 {num_test_files} 张作为测试集。")

    # 随机选择指定数量的文件
    test_files = random.sample(files, num_test_files)
    print(f"从 {source_dir} 中随机抽取 {len(test_files)} 张图片移动到 {test_dir}...")

    # 移动文件
    moved_count = 0
    for file_name in test_files:
        source_path = os.path.join(source_dir, file_name)
        destination_path = os.path.join(test_dir, file_name)
        # 确保目标文件不存在，避免意外覆盖
        if not os.path.exists(destination_path):
            shutil.move(source_path, destination_path)
            moved_count += 1
    
    print(f"成功移动 {moved_count} 张图片。")
    print("-" * 30)


if __name__ == '__main__':
    # --- 配置路径 ---
    # !! 请根据您的实际文件夹结构修改这里的相对路径 !!
    BASE_DATA_DIR = "../Data/dataset/" 

    # 真实照片域 (Domain A)
    SOURCE_A = os.path.join(BASE_DATA_DIR, "trainA")
    TEST_A = os.path.join(BASE_DATA_DIR, "testA")

    # 宫崎骏风格域 (Domain B)
    SOURCE_B = os.path.join(BASE_DATA_DIR, "trainB_ghibli")
    TEST_B = os.path.join(BASE_DATA_DIR, "testB_ghibli")
    
    # --- 开始执行 ---
    print("开始划分数据集...")
    try:
        # 划分真实照片数据集
        split_dataset(SOURCE_A, SOURCE_A, TEST_A, num_test_files=100)
        
        # 划分宫崎骏风格数据集
        split_dataset(SOURCE_B, SOURCE_B, TEST_B, num_test_files=100)
        
        print("数据集划分完成！")
    except (ValueError, FileNotFoundError) as e:
        print(f"错误: {e}")
        print("请检查您的文件夹路径是否正确，以及文件夹中是否有足够的图片。")