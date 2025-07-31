import torch
import torch.nn as nn

class CycleGANLoss(nn.Module):
    """
    CycleGAN所需的所有损失函数
    1.对抗损失(GANLoss)，使用MSELoss代替传统的BCEWithLogitsLoss
      有利于训练稳定性
      目标是让判别器对真实图像输出1，对生成的虚假图像输出0
    2.循环一致性损失(CycleConsistencyLoss)，使用L1Loss，不容易产生模糊的结果
      目标是让原始图片和循环后的图片尽可能接近
    3.身份损失(IdentityLoss)，使用L1Loss
      目标是让生成的虚假图像和原始图像尽可能接近
    """
    def __init__(self, lambda_cycle=10, lambda_identity=5, device="cuda"):
        super().__init__()
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.gan_loss = nn.MSELoss()
        self.cycle_loss = nn.L1Loss()
        self.identity_loss = nn.L1Loss()

    def forward(self, real_A, real_B, fake_A, fake_B, rec_A, rec_B):
        # 对抗损失
        gan_loss_A = self.gan_loss(fake_A, torch.ones_like(fake_A))
        gan_loss_B = self.gan_loss(fake_B, torch.ones_like(fake_B))
        
        # 循环一致性损失
        cycle_loss_A = self.cycle_loss(rec_A, real_A)
        cycle_loss_B = self.cycle_loss(rec_B, real_B)
        
        # 身份损失
        identity_loss_A = self.identity_loss(fake_A, real_A)
        identity_loss_B = self.identity_loss(fake_B, real_B)

        return {
            'gan_loss_A': gan_loss_A,
            'gan_loss_B': gan_loss_B,
            'cycle_loss_A': cycle_loss_A,
            'cycle_loss_B': cycle_loss_B,
            'identity_loss_A': identity_loss_A,
            'identity_loss_B': identity_loss_B
        }