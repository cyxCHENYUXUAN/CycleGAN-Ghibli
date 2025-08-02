import torch
import os

def save_checkpoint(models, optimizer, epoch, filename="checkpoint.pth"):
    """
    为一组模型及其共享的优化器保存检查点。
    'models'应该是一个字典，例如：{'G_A2B': model1, 'G_B2A': model2}
    """
    print(f"=> 正在保存检查点到 {filename}")
    checkpoint = {
        "epoch": epoch,
        "optimizer_state_dict": optimizer.state_dict(),
    }
    # 将每个模型的state_dict以其名称为键，添加到检查点中
    for name, model in models.items():
        checkpoint[name] = model.state_dict()
 
    torch.save(checkpoint, filename)
 
def load_checkpoint(checkpoint_file, models, optimizer, lr, device):
    """
    为一组模型及其共享的优化器加载检查点。
    'models'应该是一个字典，例如：{'G_A2B': model1, 'G_B2A': model2}
    返回需要从中继续训练的epoch序号。
    """
    if not os.path.isfile(checkpoint_file):
        print(f"=> 在 '{checkpoint_file}' 未找到检查点")
        return 0  # 如果文件不存在，从epoch 0开始
 
    print(f"=> 正在加载检查点 '{checkpoint_file}'")
    checkpoint = torch.load(checkpoint_file, map_location=device)
 
    # 从检查点中加载每个模型的权重
    for name, model in models.items():
        if name in checkpoint:
            model.load_state_dict(checkpoint[name])
        else:
            print(f"在检查点中未找到模型'{name}'的权重。")
 
    # 加载优化器状态
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # 如果需要，重置学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        print("在检查点中未找到优化器的状态")
 
    # 获取epoch序号
    start_epoch = checkpoint.get("epoch", -1) + 1
    print(f"检查点已加载，将从 epoch{start_epoch}继续训练")
    return start_epoch