import os
from PIL import Image
import warnings

# --- 配置区 ---
# 将要检查的文件夹路径放入这个列表
DIRECTORIES_TO_CHECK = [
    'Data/dataset/trainA',
    'Data/dataset/trainB_ghibli'
]

# --- 检查逻辑 ---
def check_images_in_directory(directory):
    """
    遍历指定目录中的所有图像文件，并检查它们是否可以成功加载。
    """
    print(f"\n{'='*20}")
    print(f"正在检查文件夹: {directory}")
    print(f"{'='*20}")

    if not os.path.isdir(directory):
        print(f"错误: 文件夹 '{directory}' 不存在！")
        return

    found_issues = 0
    # 获取目录下所有文件
    try:
        files = os.listdir(directory)
    except FileNotFoundError:
        print(f"错误：无法访问文件夹 '{directory}'")
        return
        
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    if not image_files:
        print("未找到任何图片文件。")
        return

    print(f"找到 {len(image_files)} 个图片文件，开始检查...")

    for filename in image_files:
        filepath = os.path.join(directory, filename)
        try:
            # 尝试打开并加载图片
            with Image.open(filepath) as img:
                # img.load() 会完整读取图像数据到内存，
                # 如果文件有问题，这一步很可能会抛出异常。
                img.load()
                
                # 特别是对于JPEG，可以检查是否有截断
                if img.format == 'JPEG' and hasattr(img, 'is_truncated') and img.is_truncated:
                    print(f"[警告] 文件可能已截断: {filepath}")
                    found_issues += 1

        except (IOError, SyntaxError) as e:
            # IOError: 文件损坏或无法读取
            # SyntaxError: Pillow 解析某些格式时可能抛出
            print(f"[严重问题] 无法加载文件: {filepath}")
            print(f"  -> 错误: {e}")
            found_issues += 1
        except Exception as e:
            # 捕获其他所有可能的异常
            print(f"[未知问题] 加载时发生意外错误: {filepath}")
            print(f"  -> 错误: {e}")
            found_issues += 1
            
    if found_issues == 0:
        print(f"\n检查完成！文件夹 '{directory}' 中的所有图片均可正常加载。")
    else:
        print(f"\n检查完成！在 '{directory}' 中共发现 {found_issues} 个有问题的图片。")


if __name__ == "__main__":
    # 配置警告过滤器，将所有UserWarning（Pillow的EXIF警告属于此类）都变成需要处理的异常
    # 这样我们就能捕获到"Possibly corrupt EXIF data"这类警告了
    warnings.simplefilter('error', UserWarning)

    print("开始进行数据集图片完整性检查...")
    
    for directory in DIRECTORIES_TO_CHECK:
        check_images_in_directory(directory)

    print("\n所有指定的文件夹都已检查完毕。")