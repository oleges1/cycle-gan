from unet import *
from datasets.cityscapes import *
from datasets.cycle import *
from discriminator import Discriminator
from torch import nn
from torch.utils.data import DataLoader
import itertools
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils import *
import os

def datasets_by_name(name, params):
    del params['name']
    if name == 'cityscapes':
        return CityScapes(**params), CityScapes('val', **params)
    elif name == 'edges2shoes':
        data_train = CycleDataset('data/edges2shoes/', 'train')
        data_test = CycleDataset('data/edges2shoes/', 'val')
        return data_train, data_test
    elif name == 'folder2folder':
        data_train = Folder2FolderDataset(params['folder_left'], params['folder_right'])
        data_test = Folder2FolderDataset(params['folder_left'], params['folder_right'], phase='val')
    else:
        raise NotImplementedError('Unknown dataset')


def train(config):
    genAB = UNet(3, 3, bilinear=config.model.bilinear_upsample).cuda()
    init_weights(genAB, 'normal')
    genBA = UNet(3, 3, bilinear=config.model.bilinear_upsample).cuda()
    init_weights(genBA, 'normal')
    discrA = Discriminator(3).cuda()
    init_weights(discrA, 'normal')
    discrB = Discriminator(3).cuda()
    init_weights(discrB, 'normal')

    writer = SummaryWriter(config.name)
    data_train, data_test = datasets_by_name(config.dataset.name, config.dataset)
    train_dataloader = DataLoader(data_train, batch_size=config.bs, shuffle=True, num_workers=config.num_workers)
    test_dataloader = DataLoader(data_test, batch_size=config.bs, shuffle=True, num_workers=config.num_workers)

    idt_loss = nn.L1Loss()
    cycle_consistency = nn.L1Loss()
    discriminator_loss = nn.BCELoss()
    lambda_idt, lambda_C, lambda_D = config.loss.lambda_idt, config.loss.lambda_C, config.loss.lambda_D

    optG = torch.optim.Adam(itertools.chain(genAB.parameters(), genBA.parameters()), lr=config.train.lr, betas=(config.train.beta1, 0.999))
    optD = torch.optim.Adam(itertools.chain(discrA.parameters(), discrB.parameters()), lr=config.train.lr, betas=(config.train.beta1, 0.999))

    for epoch in range(config.train.epochs):
        set_train([genAB, genBA, discrA, discrB])
        set_requires_grad([genAB, genBA, discrA, discrB], True)
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
                loss_D_fake, loss_D_true = 0, 0
                optD.zero_grad()
                logits = discrA(fake_A.detach())
                loss_D_fake += discriminator_loss(logits, torch.zeros_like(logits))

                logits = discrB(fake_B.detach())
                loss_D_fake += discriminator_loss(logits, torch.zeros_like(logits))
                loss_D_fake.backward()
                torch.nn.utils.clip_grad_norm_(itertools.chain(discrA.parameters(), discrB.parameters()), 15)
                optD.step()

                optD.zero_grad()
                logits = discrA(batch_A)
                loss_D_true += discriminator_loss(logits, torch.ones_like(logits))
                logits = discrB(batch_B)
                loss_D_true += discriminator_loss(logits, torch.ones_like(logits))
                loss_D_true.backward()
                torch.nn.utils.clip_grad_norm_(itertools.chain(discrA.parameters(), discrB.parameters()), 15)
                optD.step()
                loss_D = loss_D_fake + loss_D_true
            if (i % config.train.verbose_period == 0):
                writer.add_scalar('train/loss_G', loss_G.item(), len(train_dataloader) * epoch + i)
                if lambda_D > 0:
                    writer.add_scalar('train/loss_D', loss_D.item(), len(train_dataloader) * epoch + i)
                    writer.add_scalar('train/mean_D_A', discr_feedbackA.mean().item(), len(train_dataloader) * epoch + i)
                    writer.add_scalar('train/mean_D_B', discr_feedbackB.mean().item(), len(train_dataloader) * epoch + i)
                for batch_i in range(fake_A.shape[0]):
                    concat = (torch.cat([fake_A[batch_i], batch_B[batch_i]], dim=-1) + 1.) / 2.
                    writer.add_image('train/fake_A_' + str(batch_i), concat, len(train_dataloader) * epoch + i)
                for batch_i in range(fake_B.shape[0]):
                    concat = (torch.cat([fake_B[batch_i], batch_A[batch_i]], dim=-1) + 1.) / 2.
                    writer.add_image('train/fake_B_' + str(batch_i), concat, len(train_dataloader) * epoch + i)
        if not config.validate:
            continue
        set_eval([genAB, genBA, discrA, discrB])
        set_requires_grad([genAB, genBA, discrA, discrB], False)
        loss_G, loss_D, discr_feedbackA_mean, discr_feedbackB_mean = 0, 0, 0, 0
        for i, (batch_A, batch_B) in enumerate(tqdm(test_dataloader)):
            batch_A, batch_B = batch_A.cuda(), batch_B.cuda()
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
                discr_feedbackA = discrA(fake_A)
                discr_feedbackB = discrB(fake_B)
                loss_G += discriminator_loss(discr_feedbackA, torch.ones_like(discr_feedbackA)) * lambda_D
                loss_G += discriminator_loss(discr_feedbackB, torch.ones_like(discr_feedbackB)) * lambda_D
                discr_feedbackA_mean += discr_feedbackA.mean()
                discr_feedbackB_mean += discr_feedbackB.mean()
            if lambda_D > 0:
                loss_D_fake, loss_D_true = 0, 0
                logits = discrA(fake_A.detach())
                loss_D_fake += discriminator_loss(logits, torch.zeros_like(logits))
                logits = discrB(fake_B.detach())
                loss_D_fake += discriminator_loss(logits, torch.zeros_like(logits))
                logits = discrA(batch_A)
                loss_D_true += discriminator_loss(logits, torch.ones_like(logits))
                logits = discrB(batch_B)
                loss_D_true += discriminator_loss(logits, torch.ones_like(logits))
                loss_D += loss_D_fake + loss_D_true
            if i == 0:
                for batch_i in range(fake_A.shape[0]):
                    concat = (torch.cat([fake_A[batch_i], batch_B[batch_i]], dim=-1) + 1.) / 2.
                    writer.add_image('val/fake_A_' + str(batch_i), concat, epoch)
                for batch_i in range(fake_B.shape[0]):
                    concat = (torch.cat([fake_B[batch_i], batch_A[batch_i]], dim=-1) + 1.) / 2.
                    writer.add_image('val/fake_B_' + str(batch_i), concat, epoch)
        loss_G /= len(test_dataloader)
        writer.add_scalar('val/loss_G', loss_G.item(), epoch)
        if lambda_D > 0:
            loss_D /= len(test_dataloader)
            discr_feedbackA_mean /= len(test_dataloader)
            discr_feedbackB_mean /= len(test_dataloader)
            writer.add_scalar('val/loss_D', loss_D.item(), epoch)
            writer.add_scalar('val/mean_D_A', discr_feedbackA_mean.item(), epoch)
            writer.add_scalar('val/mean_D_B', discr_feedbackB_mean.item(), epoch)
        torch.save({
            'genAB': genAB.state_dict(),
            'genBA': genBA.state_dict(),
            'discrA': discrA.state_dict(),
            'discrB': discrB.state_dict(),
            'optG': optG.state_dict(),
            'optD': optD.state_dict(),
            'epoch': epoch
        }, os.path.join(config.name, 'model.pth'))
