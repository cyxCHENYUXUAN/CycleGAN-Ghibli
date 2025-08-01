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
        """
        在计算判别器损失时，使用MSELoss作为最小二乘GAN的形式，代替传统的BCEWithLogitsLoss，
        生成器的GANLoss也是同样的道理，使用MSELoss
        1.传统的BCEWithLogitsLoss的判别器的计分规则是概率值，输出范围在0到1之间，给真实图片一个接近1的概率，给虚假图片一个接近0的概率，
          因此，生成器的“欺骗目标”就是：让自己生成的图片，在经过判别器后，能得到一个接近1的概率
        2.MSELoss的判别器的计分规则是分数，给真实图片一个接近1的分数，给虚假图片一个接近0的分数，
          因此，生成器的“欺骗目标”就是：让自己生成的图片，在经过判别器后，能得到一个接近1的分数
        3.GAN的本质是一个二人零和博弈
          判别器 (D) 的目标: 尽可能好地分辨真实样本和伪造样本
          生成器 (G) 的目标: 尽可能好地欺骗判别器，让判别器把他生成的样本误判为真实样本
        """
        self.gan_loss_fn = nn.MSELoss()
        self.cycle_loss_ = nn.L1Loss()
        self.identity_loss_fn = nn.L1Loss()
        #定义标签，真实标签为1，虚假标签为0，用于计算对抗损失，反向传播更新参数时，不需要计算梯度
        #register_buffer是torch.nn.Module的一个方法，用于注册一个缓冲区，缓冲区是一个Tensor
        self.register_buffer('valid', torch.tensor(1.0, device=device))
        self.register_buffer('fake', torch.tensor(0.0, device=device))

    def calculate_G_loss(self, real_A, real_B, fake_A, fake_B, reconstructed_A, reconstructed_B, identity_A, identity_B, pred_fake_A, pred_fake_B):
        """
        pred_fake_A: 判别器D_A对生成的虚假图像fake_A的预测结果
        pred_fake_B: 判别器D_B对生成的虚假图像fake_B的预测结果
        """
        #对抗损失GANLoss
        loss_G_A2B_gan = self.gan_loss_fn(pred_fake_B, self.valid.expand_as(pred_fake_B))
        loss_G_B2A_gan = self.gan_loss_fn(pred_fake_A, self.valid.expand_as(pred_fake_A))
        #total_gan_loss不取平均对结果影响不大
        total_gan_loss = loss_G_A2B_gan + loss_G_B2A_gan

        #循环一致性损失CycleConsistencyLoss
        loss_cycle_A = self.cycle_loss_fn(reconstructed_A, real_A)
        loss_cycle_B = self.cycle_loss_fn(reconstructed_B, real_B)
        total_cycle_loss = (loss_cycle_A + loss_cycle_B) * self.lambda_cycle

        #身份损失IdentityLoss
        loss_identity_A = self.identity_loss_fn(identity_A, real_A)
        loss_identity_B = self.identity_loss_fn(identity_B, real_B)
        total_identity_loss = (loss_identity_A + loss_identity_B) * self.lambda_identity

        #总损失
        loss_G = total_gan_loss + total_cycle_loss + total_identity_loss

        loss_dict = {
            "G_total": loss_G,
            "G_gan": total_gan_loss,
            "G_cycle": total_cycle_loss,
            "G_identity": total_identity_loss
        }

        return loss_G, loss_dict
    
    def calculate_D_loss(self, pred_real, pred_fake):
        """
        pred_real: 判别器D对真实图像的预测结果
        pred_fake: 判别器D对生成的虚假图像的预测结果
        """
        #对抗损失GANLoss
        #真实损失：判别器应该对真实图像输出1
        #虚假损失：判别器应该对生成的虚假图像输出0
        loss_D_real = self.gan_loss_fn(pred_real, self.valid.expand_as(pred_real))
        loss_D_fake = self.gan_loss_fn(pred_fake, self.fake.expand_as(pred_fake))
        #总损失取平均
        #取平均遵循了CycleGAN原始论文的损失函数定义（没有使用原始GAN的交叉熵损失，采用了最小二乘GAN），同时这样做可以使判别器的训练更稳定
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        """
        L_D(D) = 0.5 * E[(D(x_real) - 1)²] + 0.5 * E[(D(x_fake) - 0)²]
        E[(D(x_real) - 1)²]是代码中的loss_D_real。它希望判别器对真实样本的输出D(x_real)尽可能接近1
        E[(D(x_fake) - 0)²]是代码中的loss_D_fake。它希望判别器对虚假样本的输出D(x_fake)尽可能接近0
        0.5是原始CycleGAN论文中对判别器损失的定义，目的是为了使损失函数的值更平滑，避免梯度爆炸
        """

        loss_dict = {
            "D_total": loss_D,
            "D_real": loss_D_real,
            "D_fake": loss_D_fake
        }

        return loss_D, loss_dict