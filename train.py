import torch
import torch.nn as nn
import torch.optim as optim
import itertools
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
from models.Generator.UNetPPGenerator import UNetPPGenerator
from models.Discriminator.PatchGAN70Discriminator import Discriminator
from utils.utils import *
from utils.ReplayBuffer import ReplayBuffer
from utils.dataset import RealGhibliDataset
from utils.loss import CycleGANLoss

def set_trainable(nets, trainable=False):
    """
    辅助函数，用于设置网络是否为可训练状态。
    trainable=False会冻结网络参数，不计算梯度。
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = trainable

def save_training_sample(epoch, loader, G_A2B, G_B2A, device, output_dir):
    """在每个epoch结束时，从训练集中随机取一个样本并保存生成结果"""
    os.makedirs(output_dir, exist_ok=True)
 
    # 从训练加载器中获取一个随机批次的数据
    batch = next(iter(loader))
    real_A = batch['real'].to(device)
    real_B = batch['ghibli'].to(device)
 
    # 将生成器设为评估模式进行推理
    G_A2B.eval()
    G_B2A.eval()
    with torch.no_grad():
        # A -> B -> A
        fake_B = G_A2B(real_A)
        reconstructed_A = G_B2A(fake_B)
        # B -> A -> B
        fake_A = G_B2A(real_B)
        reconstructed_B = G_A2B(fake_A)
 
    img_sample_A = torch.cat((real_A.data, fake_B.data, reconstructed_A.data), -1) #水平拼接
    img_sample_B = torch.cat((real_B.data, fake_A.data, reconstructed_B.data), -1)

    save_image(img_sample_A, f"{output_dir}/A_real_fake_recon_{epoch}.png", normalize=True)
    save_image(img_sample_B, f"{output_dir}/B_ghibli_fake_recon_{epoch}.png", normalize=True)
    
    # 恢复训练模式
    G_A2B.train()
    G_B2A.train()

def main():
    IMG_SIZE = 128
    BATCH_SIZE = 1
    LR = 0.0002
    EPOCHS = 200
    DEVICE = "cuda"

    #初始化模型
    G_A2B = UNetPPGenerator(in_channels=3, out_channels=3, features=[64,128,256,512]).to(DEVICE)
    G_B2A = UNetPPGenerator(in_channels=3, out_channels=3, features=[64,128,256,512]).to(DEVICE)
    D_A = Discriminator(in_channels=3).to(DEVICE)
    D_B = Discriminator(in_channels=3).to(DEVICE)

    #初始化优化器
    optimizer_G = optim.Adam(itertools.chain(G_A2B.parameters(), G_B2A.parameters()), lr=LR, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(itertools.chain(D_A.parameters(), D_B.parameters()), lr=LR, betas=(0.5, 0.999))

    # 在 100 epoch 后，学习率开始线性衰减到0
    N_EPOCHS_DECAY = EPOCHS // 2
    lr_scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda epoch: 1.0 - max(0, epoch + 1 - N_EPOCHS_DECAY) / (EPOCHS - N_EPOCHS_DECAY))
    lr_scheduler_D = optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lambda epoch: 1.0 - max(0, epoch + 1 - N_EPOCHS_DECAY) / (EPOCHS - N_EPOCHS_DECAY))

    #初始化损失函数
    loss_caculator = CycleGANLoss(lambda_cycle=10, lambda_identity=5, device=DEVICE)

    #初始化数据集
    GHIBLI_DIR = "Data/dataset/trainB_ghibli"
    REAL_DIR = "Data/dataset/trainA"
    # TEST_GHIBLI_DIR = "Data/dataset/testB_ghibli"
    # TEST_REAL_DIR = "Data/dataset/testA"
    train_transforms = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # 将图片像素值归一化到[-1, 1]
        ])
    dataset = RealGhibliDataset(ghibli_dir=GHIBLI_DIR, real_dir=REAL_DIR, transform=train_transforms)
    train_loader = DataLoader(dataset=dataset,batch_size=1,shuffle=True,num_workers=2,drop_last=True)
    # test_transforms = transforms.Compose([
    #         transforms.Resize((IMG_SIZE, IMG_SIZE)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    #     ])
    # test_dataset = RealGhibliDataset(ghibli_dir=TEST_GHIBLI_DIR, real_dir=TEST_REAL_DIR, transform=test_transforms)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)

    #初始化重放缓冲区
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()


    #加载检查点
    start_epoch = 0 #默认从0开始
    LOAD_MODEL = False  # 设置为True以加载检查点
    SAVE_MODEL = True   # 设置为True以保存检查点
    CHECKPOINT_DIR = "checkpoints"
    CHECKPOINT_GEN = "gen.pth"   # 生成器检查点文件名
    CHECKPOINT_DISC = "disc.pth" # 判别器检查点文件名
    if LOAD_MODEL:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        #定义要加载的模型字典
        gen_models_to_load = {'G_A2B': G_A2B, 'G_B2A': G_B2A}
        disc_models_to_load = {'D_A': D_A, 'D_B': D_B}
        #加载生成器组
        start_epoch = load_checkpoint(os.path.join(CHECKPOINT_DIR, CHECKPOINT_GEN), gen_models_to_load, optimizer_G, LR, DEVICE)
        #加载判别器组（不需要重新赋值start_epoch）
        load_checkpoint(os.path.join(CHECKPOINT_DIR, CHECKPOINT_DISC), disc_models_to_load, optimizer_D, LR, DEVICE)
        for i in range(start_epoch):
            #快进到正确的epoch，确保学习率是接着上次的进度衰减
            lr_scheduler_G.step()
            lr_scheduler_D.step()

    #训练循环
    print(f"开始训练，设备: {DEVICE}, 起始epoch: {start_epoch}")
    for epoch in range(start_epoch, EPOCHS):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        G_A2B.train()
        G_B2A.train()
        D_A.train()
        D_B.train()

        for i,batch in enumerate(train_loader):
            real_A = batch['real'].to(DEVICE)
            real_B = batch['ghibli'].to(DEVICE)

            #训练生成器
            set_trainable([D_A, D_B], False)
            optimizer_G.zero_grad()
            fake_B = G_A2B(real_A)
            reconstructed_A = G_B2A(fake_B)
            fake_A = G_B2A(real_B)
            reconstructed_B = G_A2B(fake_A)

            #生成器损失
            identity_A = G_B2A(real_A)
            identity_B = G_A2B(real_B)
            pred_fake_A = D_A(fake_A)
            pred_fake_B = D_B(fake_B)
            loss_G, g_loss_dict = loss_caculator.calculate_G_loss(
                real_A, real_B, fake_A, fake_B, reconstructed_A, reconstructed_B,
                identity_A, identity_B, pred_fake_A, pred_fake_B
            )
            loss_G.backward()
            optimizer_G.step()

            #训练判别器
            """
            fake_A: 刚才生成器在当前批次新的fake。
            fake_A_buffer: 一个缓冲区（Replay Buffer）。里面存放着一堆之前生成器伪造过的假照片。
            fake_A_from_buffer: 最终用来训练判别器的fake。它有50%的概率是刚才的fake_A，另外50%的概率是从fake_A_buffer里随机抽的一张旧fake。
            pred_real_A: 判别器D_A对真实图像real_A的打分。
            pred_fake_A_for_D: 判别器D_A对fake_A_from_buffer的打分。
            .detach(): 切断梯度流,在这里只希望更新鉴定师D_A的参数，而不希望梯度传回到生成这张假照片的生成器G_B2A。
            """
            set_trainable([D_A, D_B], True)
            optimizer_D.zero_grad()
            fake_A_from_buffer, fake_B_from_buffer = fake_A_buffer.push_and_pop(fake_A), fake_B_buffer.push_and_pop(fake_B)
            pred_real_A, pred_fake_A_for_D = D_A(real_A), D_A(fake_A_from_buffer.detach())
            loss_D_A, _ = loss_caculator.calculate_D_loss(pred_real_A, pred_fake_A_for_D)
            loss_D_A.backward()
            pred_real_B, pred_fake_B_for_D = D_B(real_B), D_B(fake_B_from_buffer.detach())
            loss_D_B, _ = loss_caculator.calculate_D_loss(pred_real_B, pred_fake_B_for_D)
            loss_D_B.backward()
            optimizer_D.step()

            total_d_loss = (loss_D_A.item() + loss_D_B.item()) / 2
            current_lr = optimizer_G.param_groups[0]['lr']
            progress_bar.set_postfix(
                G_Total=loss_G.item(),                       #生成器总损失
                G_GAN=g_loss_dict['G_gan'].item(),           #生成器GAN损失
                G_Cycle=g_loss_dict['G_cycle'].item(),       #生成器Cycle损失
                G_Identity=g_loss_dict['G_identity'].item(), #生成器Identity损失
                D_Total=total_d_loss,                        #判别器总损失
                D_A=loss_D_A.item(),                         #D_A损失
                D_B=loss_D_B.item(),                         #D_B损失
                lr=current_lr                                #学习率
            )

        #更新学习率
        lr_scheduler_G.step()
        lr_scheduler_D.step()

        #保存训练样本
        SAMPLE_OUTPUT_DIR = "training_samples"
        save_training_sample(epoch + 1, train_loader, G_A2B, G_B2A, DEVICE, SAMPLE_OUTPUT_DIR)

        # 每10个epoch保存一次检查点
        if SAVE_MODEL and (epoch + 1) % 10 == 0:
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            #使用字典保存模型
            gen_models_to_save = {'G_A2B': G_A2B, 'G_B2A': G_B2A}
            disc_models_to_save = {'D_A': D_A, 'D_B': D_B}
            save_checkpoint(gen_models_to_save, optimizer_G, epoch, filename=os.path.join(CHECKPOINT_DIR, CHECKPOINT_GEN))
            save_checkpoint(disc_models_to_save, optimizer_D, epoch,filename=os.path.join(CHECKPOINT_DIR, CHECKPOINT_DISC))
    print("训练完成")

if __name__ == "__main__":
    main()