import torch
import os

def save_checkpoint(model, filename='checkpoint.pth.tar'):
    print("正在保存检查点")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": model.optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)
    print("检查点已保存到", filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr, device):
    if os.path.isfile(checkpoint_file):
        print("正在加载检查点", checkpoint_file)
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print("检查点已加载")
    else:
        print("检查点不存在")