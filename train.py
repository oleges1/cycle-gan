from unet import *
from datasets.cityscapes import *
from discriminator import Discriminator
from torch import nn
from torch.utils.data import DataLoader
import itertools
from tensorboardX import SummaryWriter
from tqdm import tqdm


lambda_idt = 0
lambda_C = 0.5
lambda_D = 0.5
lr = 0.0002
beta1 = 0.5
verbose_period = 50
epochs = 100
writer = SummaryWriter("cityscapes")


def train():
    genAB = UNet(3, 3, bilinear=False).cuda()
    init_weights(genAB, 'normal')
    genBA = UNet(3, 3, bilinear=False).cuda()
    init_weights(genBA, 'normal')
    discrA = Discriminator(3).cuda()
    init_weights(discrA, 'normal')
    discrB = Discriminator(3).cuda()
    init_weights(discrB, 'normal')

    data_train = CityScapes()
    data_test = CityScapes('test')

    train_dataloader = DataLoader(data_train, batch_size=8, shuffle=True, num_workers=4)

    idt_loss = nn.L1Loss()
    cycle_consistency = nn.L1Loss()
    discriminator_loss = nn.BCELoss()

    optG = torch.optim.AdamW(itertools.chain(genAB.parameters(), genBA.parameters()), lr=lr)
    optD = torch.optim.AdamW(itertools.chain(discrA.parameters(), discrB.parameters()), lr=lr)

    for epoch in range(epochs):
        for i, (batch_A, batch_B) in enumerate(tqdm(train_dataloader)):
            batch_A, batch_B = batch_A.cuda(), batch_B.cuda()
            optG.zero_grad()
            loss_G, loss_D = 0, 0
            fake_B = genAB(batch_A)
            cycle_A = genBA(fake_B)
            fake_A = genBA(batch_B)
            cycle_B = genAB(fake_A)
            if lambda_idt > 0:
                loss_G += idt_loss(fake_B, batch_B) * lambda_idt
                loss_G += idt_loss(fake_A, batch_A) * lambda_idt
            if lambda_C > 0:
                loss_G += cycle_consistency(cycle_A, batch_A) * lambda_C
                loss_G += cycle_consistency(cycle_B, batch_B) * lambda_C
            if lambda_D > 0:
                set_requires_grad([discrA, discrB], False)
                discr_feedbackA = discrA(fake_A)
                discr_feedbackB = discrB(fake_B)
                loss_G += discriminator_loss(discr_feedbackA, torch.ones_like(discr_feedbackA)) * lambda_D
                loss_G += discriminator_loss(discr_feedbackB, torch.ones_like(discr_feedbackB)) * lambda_D
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(itertools.chain(genAB.parameters(), genBA.parameters()), 15)
            optG.step()
            if lambda_D > 0:
                set_requires_grad([discrA, discrB], True)
                optD.zero_grad()
                batch_DA = torch.cat([fake_A.detach(), batch_A])
                logits = discrA(batch_DA)
                true = torch.ones_like(logits)
                true[:fake_A.shape[0]] = 0
                loss_D += discriminator_loss(logits, true)

                batch_DB = torch.cat([fake_B.detach(), batch_B])
                logits = discrB(batch_DB)
                true = torch.ones_like(logits)
                true[:fake_B.shape[0]] = 0
                loss_D += discriminator_loss(logits, true)
                loss_D.backward()
                torch.nn.utils.clip_grad_norm_(itertools.chain(discrA.parameters(), discrB.parameters()), 15)
                optD.step()
            if (i % verbose_period == 0):
                writer.add_scalar('train/loss_G', loss_G.item(), len(train_dataloader) * epoch + i)
                if lambda_D > 0:
                    writer.add_scalar('train/loss_D', loss_D.item(), len(train_dataloader) * epoch + i)
                    writer.add_scalar('train/mean_D_A(G)', discr_feedbackA.mean().item(), len(train_dataloader) * epoch + i)
                    writer.add_scalar('train/mean_D_B(G)', discr_feedbackB.mean().item(), len(train_dataloader) * epoch + i)
                for batch_i in range(fake_A.shape[0]):
                    concat = torch.cat([fake_A[batch_i], batch_B[batch_i]], dim=-1)
                    writer.add_image('fake_A_' + str(batch_i), concat, len(train_dataloader) * epoch + i)
                for batch_i in range(fake_B.shape[0]):
                    concat = torch.cat([fake_B[batch_i], batch_A[batch_i]], dim=-1)
                    writer.add_image('fake_B_' + str(batch_i), concat, len(train_dataloader) * epoch + i)
