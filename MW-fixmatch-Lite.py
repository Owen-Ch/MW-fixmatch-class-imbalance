import random
import time
import logging
import argparse
import math
import sys
import copy
import time
import os
import datetime
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import *
from dataset.data_loader import cifar_10_data_loader, cifar_100_data_loader, stl_10_data_loader
from model import ModelEMA, VNet, MLP, build_wideresnet
from WRN28 import WideResNet
from utils import *
from itertools import cycle
from torch import optim
from torch.cuda import amp
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
from torchvision.models import AlexNet
from meta import MetaSGD, MetaAdam, Metascaler
from copy import deepcopy
from dataset.ImbalancedDatasetSampler import ImbalancedDatasetSampler

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, required=True, help='experiment name')
parser.add_argument('--data-path', default='./data', type=str, help='data path')
parser.add_argument('--save-path', default='./checkpoint', type=str, help='save path')
parser.add_argument('--dataset', default='cifar10', type=str,
                    choices=['cifar10', 'cifar100', 'stl10'], help='dataset name')
parser.add_argument('--ssl-method', default='fixmatch', type=str,
                    choices=['uda', 'fixmatch'], help='ssl method')
parser.add_argument('--loss-method', default='class_balanced', type=str,
                    choices=['class_balanced', 'cross_entropy'], help='ssl method')
parser.add_argument('--reweight-all', action='store_true', help='reweight all data')
parser.add_argument('--T', default=1, type=float,
                    help='pseudo label temperature')
parser.add_argument('--unlabeled-all', default=True, type=bool,
                    help='unlabeled include labeled')
# parser.add_argument('--meta-upsampling', default=True, type=bool,
#                     help='meta data upsampling')


parser.add_argument('--sample-l', type=int, default=1500, help='max number of labeled data in a class')
parser.add_argument('--sample-ul', type=int, default=3000, help='max number of unlabeled data in a class')
parser.add_argument('--imba-l', type=int, default=None, help='imbalanced coefficient of labeled data')
parser.add_argument('--imba-ul', type=int, default=1, help='imbalanced coefficient of unlabeled data')

parser.add_argument("--expand-labels", default=True, help="expand labels to fit eval steps")
parser.add_argument("--expand_unlabels", default=5, type=int, help="expand unlabels")
parser.add_argument("--sampling_factor", default=None, type=int, help="meta dataset sampler")
parser.add_argument('--total-steps', default=300000, type=int, help='number of total steps to run')
parser.add_argument('--eval-step', default=1000, type=int, help='number of eval steps to run')
parser.add_argument('--start-step', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--workers', default=4, type=int, help='number of workers')

parser.add_argument('--num-classes', default=10, type=int, help='number of classes')
parser.add_argument('--resize', default=32, type=int, help='resize image')
parser.add_argument('--batch-size', default=64, type=int, help='train batch size')
parser.add_argument('--teacher-dropout', default=0, type=float, help='dropout on last dense layer')
parser.add_argument('--student-dropout', default=0, type=float, help='dropout on last dense layer')
parser.add_argument('--teacher_lr', default=0.01, type=float, help='train learning late')
parser.add_argument('--student_lr', default=0.01, type=float, help='train learning late')
parser.add_argument('--momentum', default=0.9, type=float, help='SGD Momentum')
parser.add_argument('--nesterov', default=True, help='use nesterov')
parser.add_argument('--weight-decay', default=0, type=float, help='train weight decay')
parser.add_argument('--ema', default=0, type=float, help='EMA decay rate')
parser.add_argument('--warmup-steps', default=0, type=int, help='warmup steps')
parser.add_argument('--student-wait-steps', default=0, type=int, help='warmup steps')
parser.add_argument('--grad-clip', default=1e9, type=float, help='gradient norm clipping')

parser.add_argument('--resume', default='', type=str, help='path to checkpoint')
parser.add_argument('--evaluate', action='store_true', help='only evaluate model on validation set')

parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')

parser.add_argument('--label-smoothing', default=0, type=float, help='label smoothing alpha')
parser.add_argument('--mu', default=7, type=int, help='coefficient of unlabeled batch size')
parser.add_argument('--threshold', default=0.95, type=float, help='pseudo label threshold')
parser.add_argument('--temperature', default=0.95, type=float, help='pseudo label temperature')

parser.add_argument('--lambda-u', default=1, type=float, help='coefficient of unlabeled loss')
parser.add_argument('--uda-steps', default=1, type=float, help='warmup steps of lambda-u')

parser.add_argument("--randaug", nargs="+", type=int, help="use it like this. --randaug 2 10")
parser.add_argument("--amp", default=True, help="use 16-bit (mixed) precision")

parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument("--local_rank", type=int, default=-1,
                    help="For distributed training: local_rank")
parser.add_argument('--imbalanced_factor', type=int, default=None)
parser.add_argument('--meta_net_hidden_size', type=int, default=100)
parser.add_argument('--meta_net_num_layers', type=int, default=1)
parser.add_argument('--meta-lr', type=float, default=1e-5)
parser.add_argument('--meta-weight-decay', type=float, default=0.)
parser.add_argument('--optimizer', type=str, default='sgd')

flag_pbar = 0


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_wait_steps=0,
                                    num_cycles=0.5,
                                    last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_wait_steps:
            return 0.0

        if current_step < num_warmup_steps + num_wait_steps:
            return float(current_step) / float(max(1, num_warmup_steps + num_wait_steps))

        progress = float(current_step - num_warmup_steps - num_wait_steps) / \
                   float(max(1, num_training_steps - num_warmup_steps - num_wait_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_schedule_with_warmup(optimizer,
                             num_warmup_steps,
                             num_training_steps,
                             num_wait_steps=0,
                             last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_wait_steps:
            return 0.2

        if current_step < num_warmup_steps + num_wait_steps:
            return 0.2 + 0.8 * float(current_step) / float(max(1, num_warmup_steps + num_wait_steps))

        return 1

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def train_loop(args, labeled_dataloader, unlabeled_dataloader, meta_dataloader, test_dataloader,
               teacher_model, pseudo_model, avg_student_model, meta_model, criterion,
               t_optimizer, pseudo_optimizer, meta_optimizer, t_scheduler, m_scheduler, t_scaler, p_scaler, m_scaler,
               pseudo_parameters):
    logger.info("***** Running Training *****")
    logger.info(f"   Task = {args.dataset}@{args.sample_l}@{args.imba_l}@{args.sample_ul}@{args.imba_ul}")
    logger.info(f"   Total steps = {args.total_steps}")
    labeled_iter = iter(labeled_dataloader)
    unlabeled_iter = iter(unlabeled_dataloader)
    meta_iter = iter(meta_dataloader)
    # for author's code formula
    # moving_dot_product = torch.empty(1).to(args.device)
    # limit = 3.0**(0.5)  # 3 = 6 / (f_in + f_out)
    # nn.init.uniform_(moving_dot_product, -limit, limit)

    for step in range(args.start_step, args.total_steps):
        if step % args.eval_step == 0:
            if step > args.stop_step:
                break
            if flag_pbar == 1:
                pbar = tqdm(range(args.eval_step), disable=args.local_rank not in [-1, 0])
            batch_time = AverageMeter()
            data_time = AverageMeter()
            s_losses = AverageMeter()
            t_losses = AverageMeter()
            t_losses_l = AverageMeter()
            t_losses_u = AverageMeter()
            t_losses_mpl = AverageMeter()
            m_losses = AverageMeter()
            mean_mask = AverageMeter()
        pseudo_model = WideResNet(num_classes=args.num_classes,
                                  depth=args.depth,
                              widen_factor=args.widen_factor,
                                  dropout=0).to(args.device)
        pseudo_model.load_state_dict(teacher_model.state_dict())
        optim_group = t_optimizer.param_groups[0]
        if args.optim == 1:
            pseudo_optimizer = optim.Adam(pseudo_model.parameters(),
                            lr=optim_group["lr"])
        else :
            pseudo_optimizer = optim.SGD(pseudo_model.parameters(),
                                     lr=optim_group["lr"],
                                     weight_decay=optim_group["weight_decay"],
                                     momentum=optim_group['momentum'],
                                     dampening=optim_group['dampening'],
                                     nesterov=optim_group['nesterov'])
        teacher_model.train()

        pseudo_model.train()

        meta_model.train()

        end = time.time()
        try:
            # error occurs ↓
            # images_l, targets = labeled_iter.next()
            images_l, targets = next(labeled_iter)
        except:
            labeled_iter = iter(labeled_dataloader)
            # error occurs ↓
            # images_l, targets = labeled_iter.next()
            images_l, targets = next(labeled_iter)

        try:
            # error occurs ↓
            # (images_uw, images_us), _ = unlabeled_iter.next()

            (images_uw, images_us), _ = next(unlabeled_iter)
        except:
            unlabeled_iter = iter(unlabeled_dataloader)
            # error occurs ↓
            # (images_uw, images_us), _ = unlabeled_iter.next()
            (images_uw, images_us), _ = next(unlabeled_iter)
        try:
            # error occurs ↓
            # images_l, targets = labeled_iter.next()
            images_m, targets_m = next(meta_iter)
        except:
            meta_iter = iter(meta_dataloader)
            # error occurs ↓
            # images_l, targets = labeled_iter.next()
            images_m, targets_m = next(meta_iter)
        data_time.update(time.time() - end)

        images_l = images_l.to(args.device)
        images_uw = images_uw.to(args.device)
        images_us = images_us.to(args.device)
        targets = targets.to(args.device)
        images_m, targets_m = images_m.to(args.device), targets_m.to(args.device)

        with amp.autocast(enabled=args.amp):
            batch_size = images_l.shape[0]
            p_images = torch.cat((images_l, images_uw, images_us))
            p_logits = pseudo_model(p_images)
            p_logits_l = p_logits[:batch_size]
            p_logits_uw, p_logits_us = p_logits[batch_size:].chunk(2)
            del p_logits
            # with torch.no_grad():
            #     m_logits = pseudo_model(images_m)
            m_logits = pseudo_model(images_m)
            m_loss_old = F.cross_entropy(m_logits.detach(), targets_m.long())
            p_loss_l_vector = criterion(p_logits_l, targets.long(), reduction='none')
            p_soft_pseudo_label = torch.softmax(p_logits_uw.detach() / args.temperature, dim=-1)
            p_max_probs, p_hard_pseudo_label = torch.max(p_soft_pseudo_label, dim=-1)
            p_mask = p_max_probs.ge(args.threshold).float()
            p_loss_u_vector = F.cross_entropy(p_logits_us, p_hard_pseudo_label,
                                              reduction='none') * p_mask
            p_loss_u_reshape = torch.reshape(p_loss_u_vector, (-1, 1))

            if args.reweight_all:  # 全加权
                p_loss_l_reshape = torch.reshape(p_loss_l_vector, (-1, 1))
                weight_input = torch.cat((p_loss_l_reshape, p_loss_u_reshape)).clone().detach()
                p_weight_l = meta_model(p_loss_l_reshape)
                p_weight_u = meta_model(p_loss_u_reshape)
                p_loss_l = torch.mean(p_weight_l * p_loss_l_reshape)
                p_loss_u = torch.mean(p_weight_u * p_loss_u_reshape)
            else:  # 只加权无监督样本
                p_loss_l = torch.mean(p_loss_l_vector)
                weight_input = p_loss_u_reshape.clone().detach()
                p_weight = meta_model(p_loss_u_reshape)

                p_loss_u = torch.mean(p_weight * p_loss_u_reshape)
            p_loss_fixmatch = p_loss_l + args.lambda_u * p_loss_u
        p_scaler.scale(p_loss_fixmatch).backward()
        if args.grad_clip > 0:
            p_scaler.unscale_(pseudo_optimizer)
            nn.utils.clip_grad_norm_(pseudo_model.parameters(), args.grad_clip)
        p_scaler.step(pseudo_optimizer)
        p_scaler.update()

        weight_out = meta_model(weight_input)
        with amp.autocast(enabled=args.amp):
            with torch.no_grad():
                meta_outputs = pseudo_model(images_m)
            m_loss_new = F.cross_entropy(meta_outputs.detach(), targets_m.long())
        # meta_outputs = pseudo_model(images_m)
        dot_product = m_loss_new - m_loss_old
        p_scaler.scale(dot_product)
        # dot_product = m_loss_old - m_loss_new
        # meta_loss = dot_product * (p_weight_l.mean() + p_weight_u.mean())
        meta_loss = torch.mean(dot_product * weight_out)
        meta_optimizer.zero_grad()
        meta_loss.backward()
        meta_optimizer.step()
        # m_scaler.scale(meta_loss).backward()
        # if args.grad_clip > 0:
        #     m_scaler.unscale_(meta_optimizer)
        #     nn.utils.clip_grad_norm_(meta_model.parameters(), args.grad_clip)
        # m_scaler.step(meta_optimizer)
        # m_scaler.update()
        #m_scheduler.step()
        # with amp.autocast(enabled=args.amp):
        #     images_m, targets_m = images_m.to(args.device), targets_m.to(args.device)
        #     meta_outputs = pseudo_model(images_m)
        #     meta_loss = F.cross_entropy(meta_outputs, targets_m.long())
        # m_scaler.scale(meta_loss).backward()
        # if args.grad_clip > 0:
        #     m_scaler.unscale_(meta_optimizer)
        #     nn.utils.clip_grad_norm_(pseudo_model.parameters(), args.grad_clip)
        #     nn.utils.clip_grad_norm_(meta_model.parameters(), args.grad_clip)
        # m_scaler.step(meta_optimizer)
        # m_scaler.update()
        # m_scheduler.step()

        with amp.autocast(enabled=args.amp):

            t_logits = teacher_model(p_images)
            t_logits_l = t_logits[:batch_size]
            t_logits_uw, t_logits_us = t_logits[batch_size:].chunk(2)
            del t_logits
            t_loss_l_vector = criterion(t_logits_l, targets.long(), reduction='none')
            soft_pseudo_label = torch.softmax(t_logits_uw.detach() / args.temperature, dim=-1)
            max_probs, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()
            # weight_u = args.lambda_u * min(1., (step + 1) / args.uda_steps)
            t_loss_u_vector = (F.cross_entropy(t_logits_us, hard_pseudo_label, reduction='none') * mask)
            t_loss_u_reshape = torch.reshape(t_loss_u_vector, (-1, 1))
            # if args.reweight_all:
            #     t_loss_l_reshape = torch.reshape(t_loss_l_vector, (-1, 1))
            #     t_loss_reshape = torch.cat((t_loss_l_reshape, t_loss_u_reshape))
            #     t_weight = meta_model(t_loss_reshape)
            #     t_loss = t_weight * t_loss_reshape
            #     t_loss_l = t_loss[:batch_size].mean()
            #     t_loss_u = t_loss[batch_size:].mean()
            # else:
            #     t_loss_l = t_loss_l_vector.mean()
            #     t_weight = meta_model(t_loss_u_reshape)
            #     t_loss_u = torch.mean(t_weight * t_loss_u_reshape)
            if args.reweight_all:  # 全加权
                t_loss_l_reshape = torch.reshape(t_loss_l_vector, (-1, 1))
                with torch.no_grad():
                    t_weight_l = meta_model(t_loss_l_reshape.data)
                    t_weight_u = meta_model(t_loss_u_reshape.data)
                t_loss_l = torch.mean(t_weight_l * t_loss_l_reshape)
                t_loss_u = torch.mean(t_weight_u * t_loss_u_reshape)
            else:  # 只加权无监督样本
                t_loss_l = torch.mean(t_loss_l_vector)
                with torch.no_grad():
                    t_weight = meta_model(t_loss_u_reshape.data)
                t_loss_u = torch.mean(t_weight * t_loss_u_reshape)
            t_loss_fixmatch = t_loss_l + args.lambda_u * t_loss_u
        t_scaler.scale(t_loss_fixmatch).backward()
        if args.grad_clip > 0:
            t_scaler.unscale_(t_optimizer)
            nn.utils.clip_grad_norm_(teacher_model.parameters(), args.grad_clip)
        t_scaler.step(t_optimizer)
        t_scaler.update()
        t_scheduler.step()

        if args.ema > 0:
            avg_student_model.update_parameters(teacher_model)

        teacher_model.zero_grad()
        # meta_model.zero_grad()
        # pseudo_model.zero_grad()

        t_losses_l.update(t_loss_l.item())
        t_losses_u.update(t_loss_u.item())
        t_losses.update(t_loss_fixmatch.item())
        m_losses.update(meta_loss.item())
        mean_mask.update(mask.mean().item())
        batch_time.update(time.time() - end)
        print("{:.4f}".format(batch_time.avg))
        if flag_pbar == 1:
            pbar.set_description(
                f"Train Iter: {step + 1:3}/{args.total_steps:3}. "
                f"LR: {get_lr(t_optimizer):.4f}. Data: {data_time.avg:.2f}s. "
                f"Batch: {batch_time.avg:.4f}s. "
                f"T_Loss: {t_losses.avg:.4f}. Mask: {mean_mask.avg:.4f}. ")
            pbar.update()
        if (step + 1) % 50 == 0:
            logger.info(f"Train Iter: {step + 1:3}/{args.total_steps:3}. "
                        f"LR: {get_lr(t_optimizer):.4f}. Data: {data_time.avg:.2f}s. "
                        f"Batch: {batch_time.avg:.2f}s. "
                        f"T_Loss: {t_losses.avg:.4f}. Mask: {mean_mask.avg:.4f}. m_loss: {m_losses.avg:.4f}. ")
        if args.local_rank in [-1, 0]:
            args.writer.add_scalar("lr", get_lr(t_optimizer), step)
            args.writer.add_scalar("meta_lr", get_lr(meta_optimizer), step)

        args.num_eval = step // args.eval_step
        if (step + 1) % args.eval_step == 0:
            if flag_pbar == 1:
                pbar.close()
            if args.local_rank in [-1, 0]:

                args.writer.add_scalar("train/3.t_labeled", t_losses_l.avg, args.num_eval)
                args.writer.add_scalar("train/4.t_unlabeled", t_losses_u.avg, args.num_eval)
                args.writer.add_scalar("train/4.t_loss", t_losses.avg, args.num_eval)
                args.writer.add_scalar("train/4.m_loss", m_losses.avg, args.num_eval)
                args.writer.add_scalar("train/6.mask", mean_mask.avg, args.num_eval)
                #                 wandb.log({"train/1.s_loss": s_losses.avg,
                #                            "train/2.t_loss": t_losses.avg,
                #                            "train/3.t_labeled": t_losses_l.avg,
                #                            "train/4.t_unlabeled": t_losses_u.avg,
                #                            "train/5.t_mpl": t_losses_mpl.avg,
                #                            "train/6.mask": mean_mask.avg})

                test_model = avg_student_model if avg_student_model is not None else teacher_model
                test_loss, top1, top5, gm = evaluate(args, test_dataloader, teacher_model, criterion)
                class_names = test_dataloader.dataset.classes
                test_loss_ema, top1_ema, top5_ema, gm_ema = evaluate(args, test_dataloader, test_model,
                                                                     criterion)
                # add_confusion_matrix(args.writer, cmtx_ema, num_classes=len(class_names), global_step=args.num_eval,
                #                      class_names=class_names,
                #                      tag="Train Confusion Matrix", figsize=[10, 8])

                args.writer.add_scalar("test/loss", test_loss, args.num_eval)
                args.writer.add_scalar("test/acc@1", top1, args.num_eval)
                args.writer.add_scalar("test/GM", gm, args.num_eval)
                args.writer.add_scalar("ema/loss", test_loss_ema, args.num_eval)
                args.writer.add_scalar("ema/acc@1", top1_ema, args.num_eval)
                args.writer.add_scalar("ema/GM", gm_ema, args.num_eval)

                is_best = top1 > args.best_top1
                is_best_ema = top1_ema > args.best_top1_ema
                is_best_gm = gm_ema > args.best_GM
                if is_best:
                    args.best_top1 = top1
                    args.best_top5 = top5

                if is_best_ema:
                    args.best_top1_ema = top1_ema
                    args.best_top5_ema = top5_ema
                    #args.best_GM = gm_ema
                if is_best_gm:
                    args.best_GM = gm_ema

                logger.info(f"top-1 acc: {top1:.2f} ."
                            f"ema GM: {gm:.4f}")
                logger.info(f"Best top-1 acc: {args.best_top1:.2f}")
                logger.info(f"ema top-1 acc: {top1_ema:.2f} ."
                            f"ema GM: {gm_ema:.4f}")
                logger.info(f"Best ema top-1 acc: {args.best_top1_ema:.2f}")
                logger.info(f"Best ema GM: {args.best_GM:.4f}")
                save_checkpoint(args, {
                    'step': step + 1,
                    'teacher_state_dict': teacher_model.state_dict(),
                    'meta_state_dict': meta_model.state_dict(),
                    'avg_state_dict': avg_student_model.state_dict() if avg_student_model is not None else None,
                    'best_top1': args.best_top1,
                    'best_top5': args.best_top5,
                    'best_top1_ema': args.best_top1_ema,
                    'best_top5_ema': args.best_top5_ema,
                    'best_GM_ema': args.best_GM,
                    'teacher_optimizer': t_optimizer.state_dict(),
                    'meta_optimizer': meta_optimizer.state_dict(),
                    'teacher_scheduler': t_scheduler.state_dict(),
                    'teacher_scaler': t_scaler.state_dict(),
                    'meta_scaler': m_scaler.state_dict(),
                }, is_best, is_best_ema)

            if args.local_rank in [-1, 0]:
                args.writer.add_scalar("result/test_acc@1", args.best_top1)
                args.writer.add_scalar("result/test_acc@1_ema", args.best_top1_ema)
            #         wandb.log({"result/test_acc@1": args.best_top1})

    return


def evaluate(args, test_loader, model, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    class_names = test_loader.dataset.classes
    num_classes = len(class_names)
    classwise_correct = torch.zeros(num_classes).to(args.device)
    classwise_num = torch.zeros(num_classes).to(args.device)
    section_acc = torch.zeros(3)
    preds = []
    labels = []
    # test_iter = tqdm(test_loader, disable=args.local_rank not in [-1, 0])
    with torch.no_grad():
        end = time.time()
        for step, (images, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            batch_size = images.shape[0]
            images = images.to(args.device)
            targets = targets.to(args.device)
            with amp.autocast(enabled=args.amp):

                outputs = model(images)
                loss = criterion(outputs, targets)
            acc1, acc5 = accuracy(outputs, targets, (1, 5))
            losses.update(loss.item(), batch_size)
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)
            pred_label = outputs.max(1)[1]
            pred_mask = (targets == pred_label).float()
            preds.append(outputs)
            labels.append(targets)

            for i in range(num_classes):
                class_mask = (targets == i).float()
                # print(class_mask.is_cuda, pred_mask.is_cuda)
                correct_sum = (class_mask * pred_mask).sum()
                # print(correct_sum.is_cuda)
                classwise_correct[i] += correct_sum
                classwise_num[i] += class_mask.sum()

            # pred_label = outputs.max(1)[1].cpu()
            # pred_mask = (targets.cpu() == pred_label).float()
            # preds.append(outputs.cpu())
            # labels.append(targets.cpu())
            # for i in range(num_classes):
            #     class_mask = (targets.cpu() == i).float()
            #
            #     classwise_correct[i] += (class_mask * pred_mask).sum()
            #     classwise_num[i] += class_mask.sum()
            batch_time.update(time.time() - end)
            end = time.time()
            # test_iter.set_description(
            #     f"Test Iter: {step + 1:3}/{len(test_loader):3}. Data: {data_time.avg:.2f}s. "
            #     f"Batch: {batch_time.avg:.2f}s. Loss: {losses.avg:.4f}. "
            #     f"top1: {top1.avg:.2f}. top5: {top5.avg:.2f}. ")

        # test_iter.close()

        section_num = int(num_classes / 3)
        classwise_acc = (classwise_correct / classwise_num)
        section_acc[0] = classwise_acc[:section_num].mean()
        section_acc[2] = classwise_acc[-1 * section_num:].mean()
        section_acc[1] = classwise_acc[section_num:-1 * section_num].mean()
        # preds = torch.cat(preds, dim=0)
        # preds_1 = preds.max(1)[1]
        # labels = torch.cat(labels, dim=0)
        # cmtx = get_confusion_matrix(preds, labels, len(class_names))
        # if args.evaluate:
        #     print(classification_report(labels, preds_1, target_names=class_names))
        GM = 1.0
        for i in range(num_classes):
            if classwise_acc[i] == 0:
                # To prevent the N/A values, we set the minimum value as 0.001
                GM *= (1 / (100 * num_classes)) ** (1 / num_classes)
            else:
                GM *= (classwise_acc[i]) ** (1 / num_classes)
        return losses.avg, top1.avg, top5.avg, GM


def main():
    args = parser.parse_args()
    args.best_top1 = 0.
    args.best_top5 = 0.
    args.best_top1_ema = 0.
    args.best_top5_ema = 0.
    args.best_GM = 0.
    args.gpu = 0
    args.stop_step = 90000
    if args.optimizer=='adam':
        args.optim=1
    else:
        args.optim=0          
    args.world_size = 1
    args.device = torch.device('cuda', args.gpu)
    torch.cuda.empty_cache()

    log_folder = 'logs'
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    # 获取当前日期时间
    current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    # 构建日志文件路径
    log_filepath = os.path.join(log_folder,
                                f"fangfa2_{args.dataset}_{args.sample_l}_{args.imba_l}_{args.sample_ul}_{args.imba_ul}_{current_datetime}.log")
    logging.basicConfig(
        filename=log_filepath,
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARNING)
    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}")

    logger.info(dict(args._get_kwargs()))

    if args.local_rank in [-1, 0]:
        args.writer = SummaryWriter(f"results/{args.name}")

    if args.seed is not None:
        set_seed(args)

    if args.dataset == "cifar10":
        labeled_dataset, unlabeled_dataset, meta_dataset, test_dataset, num_classes, imba_l_list = cifar_10_data_loader(
            args)
    elif args.dataset == 'cifar100':
        labeled_dataset, unlabeled_dataset, meta_dataset, test_dataset, num_classes, imba_l_list = cifar_100_data_loader(
            args)
        args.stop_step = 150000
    elif args.dataset == 'stl10':
        labeled_dataset, unlabeled_dataset, meta_dataset, test_dataset, num_classes, imba_l_list = stl_10_data_loader(
            args)
        args.stop_step = 150000
    else:
        return
    train_sampler = RandomSampler
    balanced_sampler = ImbalancedDatasetSampler
    labeled_loader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=True)
    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size * args.mu,
        num_workers=args.workers,
        drop_last=True)
    meta_loader = DataLoader(
        meta_dataset,
        sampler=balanced_sampler(meta_dataset, shuffle=False, sampling_factor=args.sampling_factor),
        batch_size=args.batch_size * 2,
        num_workers=args.workers,
        drop_last=True)
    test_loader = DataLoader(test_dataset,
                             sampler=SequentialSampler(test_dataset),
                             batch_size=args.batch_size,
                             num_workers=args.workers)
    if args.dataset == "cifar10":
        args.depth, args.widen_factor = 28, 2
    elif args.dataset == 'cifar100':
        args.depth, args.widen_factor = 28, 2
    elif args.dataset == 'stl10':
        args.depth, args.widen_factor = 28, 2

    else :
        args.depth, args.widen_factor = 28, 2
    # teacher_model = build_wideresnet(num_classes=num_classes,
    #                            depth=depth,
    #                            widen_factor=widen_factor,
    #                            dropout=0)
    # pseudo_model = build_wideresnet(num_classes=num_classes,
    #                           depth=depth,
    #                           widen_factor=widen_factor,
    #                           dropout=0)
    teacher_model = WideResNet(num_classes=num_classes,
                               depth=args.depth,
                               widen_factor=args.widen_factor,
                               dropout=0)
    pseudo_model = WideResNet(num_classes=num_classes,
                              depth=args.depth,
                              widen_factor=args.widen_factor,
                              dropout=0)
    # teacher_model = WideResNet(num_classes=num_classes,
    #                            depth=depth,
    #                            widen_factor=widen_factor,
    #                            dropout=0,
    #                            dense_dropout=args.teacher_dropout)
    # pseudo_model = WideResNet(num_classes=num_classes,
    #                           depth=depth,
    #                           widen_factor=widen_factor,
    #                           dropout=0,
    #                           dense_dropout=args.student_dropout)
    # meta_model = VNet(input=1, hidden=args.meta_net_hidden_size, output=args.meta_net_num_layers)
    meta_model = MLP(hidden_size=args.meta_net_hidden_size, num_layers=args.meta_net_num_layers)

    logger.info(f"Model: WideResNet {args.depth}x{args.widen_factor}")
    logger.info(f"Params: {sum(p.numel() for p in teacher_model.parameters()) / 1e6:.2f}M")

    teacher_model.to(args.device)
    pseudo_model.to(args.device)
    meta_model.to(args.device)
    avg_student_model = None
    if args.ema > 0:
        avg_student_model = ModelEMA(teacher_model, args.ema)

    # criterion = create_loss_fn(args)
    if args.loss_method == 'class_balanced':
        criterion = Loss(loss_type="cross_entropy",beta=0.999,
                         samples_per_class=imba_l_list,
                         class_balanced=True)
    else:
        criterion = Loss(loss_type="cross_entropy")
    no_decay = ['bn']
    teacher_parameters = [
        {'params': [p for n, p in teacher_model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in teacher_model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    pseudo_parameters = [
        {'params': [p for n, p in pseudo_model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in pseudo_model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.optim == 1:
        t_optimizer = optim.Adam(teacher_parameters,
                                
                            lr=args.teacher_lr)
        pseudo_optimizer = optim.Adam(pseudo_model.parameters(),
                            lr=args.teacher_lr)
    else:
        t_optimizer = optim.SGD(teacher_parameters,
                                 lr=args.teacher_lr,
                                 momentum=args.momentum,
                                 nesterov=args.nesterov)
        pseudo_optimizer = optim.SGD(pseudo_model.parameters(),
                                 lr=args.teacher_lr,
                                 momentum=args.momentum,
                                 nesterov=args.nesterov)
            
    optim_group = t_optimizer.param_groups[0]
    
    meta_optimizer = optim.SGD(meta_model.parameters(),
                               lr=args.meta_lr,
                               weight_decay=args.meta_weight_decay)
    # meta_optimizer = optim.SGD(meta_model.parameters(),
    #                            lr=args.meta_lr,
    #                            momentum=args.momentum,
    #                            nesterov=args.nesterov,
    #                            weight_decay=args.meta_weight_decay)

    # meta_optimizer = torch.optim.Adam(meta_model.parameters(), lr=args.meta_lr, weight_decay=args.meta_weight_decay)

    t_scheduler = get_cosine_schedule_with_warmup(t_optimizer,
                                                  args.warmup_steps,
                                                  args.total_steps)
    # m_scheduler = get_cosine_schedule_with_warmup(meta_optimizer,
    #                                               args.warmup_steps,
    #                                               args.total_steps)
    m_scheduler = get_schedule_with_warmup(meta_optimizer,
                                           60000,
                                           args.total_steps)

    t_scaler = amp.GradScaler(enabled=args.amp)
    p_scaler = amp.GradScaler(enabled=args.amp)
    # p_scaler = Metascaler(enabled=args.amp)
    m_scaler = amp.GradScaler(enabled=args.amp)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"=> loading checkpoint '{args.resume}'")
            loc = f'cuda:{args.gpu}'
            checkpoint = torch.load(args.resume, map_location=loc)
            args.best_top1 = checkpoint['best_top1'].to(torch.device('cpu'))
            args.best_top5 = checkpoint['best_top5'].to(torch.device('cpu'))
            if 'best_top1_ema' in checkpoint:
                args.best_top1_ema = checkpoint['best_top1_ema'].to(torch.device('cpu'))
                args.best_top5_ema = checkpoint['best_top5_ema'].to(torch.device('cpu'))
            if not (args.evaluate):
                args.start_step = checkpoint['step']
                t_optimizer.load_state_dict(checkpoint['teacher_optimizer'])
                meta_optimizer.load_state_dict(checkpoint['meta_optimizer'])
                t_scheduler.load_state_dict(checkpoint['teacher_scheduler'])
                t_scaler.load_state_dict(checkpoint['teacher_scaler'])
                m_scaler.load_state_dict(checkpoint['meta_scaler'])
                model_load_state_dict(teacher_model, checkpoint['teacher_state_dict'])
                model_load_state_dict(meta_model, checkpoint['meta_state_dict'])
                if avg_student_model is not None:
                    model_load_state_dict(avg_student_model, checkpoint['avg_state_dict'])
            else:
                if checkpoint['avg_state_dict'] is not None:
                    model_load_state_dict(avg_student_model, checkpoint['avg_state_dict'])

                model_load_state_dict(teacher_model, checkpoint['teacher_state_dict'])

            logger.info(f"=> loaded checkpoint '{args.resume}' (step {checkpoint['step']})")
        else:
            logger.info(f"=> no checkpoint found at '{args.resume}'")

    if args.evaluate:
        test_loss, top1, top5, gm = evaluate(args, test_loader, teacher_model, criterion)
        test_loss_ema, top1_ema, top5_ema, gm_ema = evaluate(args, test_loader, avg_student_model, criterion)
        logger.info(f"top-1 acc: {top1:.2f} ."
                    f"ema GM: {gm:.4f}")
        logger.info(f"ema top-1 acc: {top1_ema:.2f} ."
                    f"ema GM: {gm_ema:.4f}")
        # test_loss, top1, top5, gm, cmtx = evaluate(args, test_loader, teacher_model, criterion)
        # test_loss_ema, top1_ema, top5_ema, gm_ema,cmtx_ema = evaluate(args, test_loader, avg_student_model, criterion)
        # class_names = test_loader.dataset.classes
        # add_confusion_matrix(args.writer, cmtx_ema, num_classes=len(class_names), global_step=args.num_eval,
        #                      class_names=class_names,
        #                      tag="Train Confusion Matrix", figsize=[10, 8])
        # img = plot_confusion_matrix(cmtx_ema, num_classes=len(class_names), class_names=class_names, figsize=[10, 8])
        #
        # img.show()
        # img1 = plot_confusion_matrix(cmtx, num_classes=len(class_names), class_names=class_names, figsize=[10, 8])
        # img1.show()
        return

    teacher_model.zero_grad()
    pseudo_model.zero_grad()
    meta_model.zero_grad()
    train_loop(args, labeled_loader, unlabeled_loader, meta_loader, test_loader,
               teacher_model, pseudo_model, avg_student_model, meta_model, criterion,
               t_optimizer, pseudo_optimizer, meta_optimizer, t_scheduler, m_scheduler, t_scaler, p_scaler, m_scaler,
               pseudo_parameters)
    return


if __name__ == '__main__':
    main()
