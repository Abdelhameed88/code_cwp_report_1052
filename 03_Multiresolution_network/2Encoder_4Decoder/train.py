import os
import sys
import time
import datetime
import json
import csv

import torch
from torch import nn
from torch.utils.data import RandomSampler, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.transforms import Compose

import utils
import network
from dataset import FWIDataset
from scheduler import WarmupMultiStepLR
import transforms as t
import csv
import psutil
import numpy as np
import random

# torch.manual_seed(12345)

# np.random.seed(12345)
# random.seed(12345)

# Function to get CPU memory usage in GB
def get_cpu_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 3)  # Convert bytes to GB

# Function to get GPU memory usage in GB
def get_gpu_memory_usage():
    gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert bytes to GB
    gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # Convert bytes to GB
    return gpu_memory_allocated, gpu_memory_reserved

def update_csv(filename, epoch, loss, loss_g1_lb1, loss_g1_lb2, loss_g1_lb3, loss_g1_lb4, \
                                         loss_g2_lb1, loss_g2_lb2, loss_g2_lb3, loss_g2_lb4, \
                             train_loss, train_loss_g1_lb1, train_loss_g1_lb2, train_loss_g1_lb3, train_loss_g1_lb4, \
                                         train_loss_g2_lb1, train_loss_g2_lb2, train_loss_g2_lb3, train_loss_g2_lb4):
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["Epoch", "val_loss", "val_loss_g1_lb1", "val_loss_g1_lb2", "val_loss_g1_lb3", "val_loss_g1_lb4", \
                                                  "val_loss_g2_lb1", "val_loss_g2_lb2", "val_loss_g2_lb3", "val_loss_g2_lb4", \
                             "tr_loss", "tr_loss_g1_lb1", "tr_loss_g1_lb2", "tr_loss_g1_lb3", "tr_loss_g1_lb4", \
                                        "tr_loss_g2_lb1", "tr_loss_g2_lb2", "tr_loss_g2_lb3", "tr_loss_g2_lb4"])
        writer.writerow([epoch, loss, loss_g1_lb1, loss_g1_lb2, loss_g1_lb3, loss_g1_lb4, \
                                      loss_g2_lb1, loss_g2_lb2, loss_g2_lb3, loss_g2_lb4, \
                         train_loss, train_loss_g1_lb1, train_loss_g1_lb2, train_loss_g1_lb3, train_loss_g1_lb4, \
                                     train_loss_g2_lb1, train_loss_g2_lb2, train_loss_g2_lb3, train_loss_g2_lb4])
def log_memory_to_csv(filename, epoch, avg_cpu, avg_gpu_allocated, avg_gpu_reserved, min_cpu, max_cpu, min_gpu_allocated, max_gpu_allocated, min_gpu_reserved, max_gpu_reserved):
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["Epoch", "Avg CPU Memory (GB)", "Avg GPU Allocated (GB)", "Avg GPU Reserved (GB), Min CPU Memory (GB)", "Max CPU Memory (GB)", "Min GPU Allocated (GB)", "Max GPU Allocated (GB)", "Min GPU Reserved (GB)", "Max GPU Reserved (GB)"])
        writer.writerow([epoch, avg_cpu, avg_gpu_allocated, avg_gpu_reserved, min_cpu, max_cpu, min_gpu_allocated, max_gpu_allocated, min_gpu_reserved, max_gpu_reserved])

step = 0

def train_one_epoch(model, criterion, optimizer, lr_scheduler, dataloader, device, epoch, print_freq, writer, args):
    global step

    # Set model to train mode
    model.train()

    # Collect memory stats
    epoch_cpu_memory = []
    epoch_gpu_allocated = []
    epoch_gpu_reserved = []

    metric_logger = utils.MetricLogger(delimiter='  ')

    # Adding 'lr'(learning rate) and 'samples/s' meters to the logger
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('samples/s', utils.SmoothedValue(window_size=10, fmt='{value:.3f}'))

    # Header for the logger
    header = 'Epoch: [{}]'.format(epoch)

    # Monitor memory usage at the start of the epoch
    if args.enable_profiling:
        cpu_usage_start = get_cpu_memory_usage()
        gpu_usage_start = get_gpu_memory_usage()
        print(f"Epoch {epoch} Start - CPU Memory: {cpu_usage_start:.4f} GB | "
              f"GPU Memory Allocated: {gpu_usage_start[0]:.4f} GB | "
              f"GPU Memory Reserved: {gpu_usage_start[1]:.4f} GB")

    # Iterating over the dataloader
    for data_Vcomp, data_Hcomp, label1, label2, label3, label4  in metric_logger.log_every(dataloader, print_freq, header):
        start_time = time.time()
        optimizer.zero_grad()
        data_Vcomp, data_Hcomp = data_Vcomp.to(device), data_Hcomp.to(device)
        
        label1, label2, label3, label4 = label1.to(device), label2.to(device), label3.to(device), label4.to(device)
        
        output_lb1, output_lb2, output_lb3, output_lb4  = model(data_Vcomp, data_Hcomp)

        loss, loss_g1_lb1, loss_g1_lb2, loss_g1_lb3, loss_g1_lb4, \
              loss_g2_lb1, loss_g2_lb2, loss_g2_lb3, loss_g2_lb4 = criterion(output_lb1, output_lb2, output_lb3, output_lb4, \
                                                                            label1, label2, label3, label4)
        
        # Collect memory stats per batch
        if args.enable_profiling:
            cpu_usage_before_back = get_cpu_memory_usage()
            gpu_usage_before_back = get_gpu_memory_usage()
            print(f"Before Backward - CPU: {cpu_usage_before_back:.4f} GB | "
                  f"GPU Allocated: {gpu_usage_before_back[0]:.4f} GB | "
                  f"GPU Reserved: {gpu_usage_before_back[1]:.4f} GB")
            epoch_cpu_memory.append(cpu_usage_before_back)
            epoch_gpu_allocated.append(gpu_usage_before_back[0])
            epoch_gpu_reserved.append(gpu_usage_before_back[1])

        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        loss_g1_val_lb1, loss_g1_val_lb2, loss_g1_val_lb3, loss_g1_val_lb4 = loss_g1_lb1.item(), loss_g1_lb2.item(), loss_g1_lb3.item(), loss_g1_lb4.item()
        loss_g2_val_lb1, loss_g2_val_lb2, loss_g2_val_lb3, loss_g2_val_lb4 = loss_g2_lb1.item(), loss_g2_lb2.item(), loss_g2_lb3.item(), loss_g2_lb4.item()
        batch_size = data_Vcomp.shape[0]
        
        metric_logger.update(loss=loss_val, loss_g1_lb1 = loss_g1_val_lb1, loss_g1_lb2 = loss_g1_val_lb2, loss_g1_lb3 = loss_g1_val_lb3, loss_g1_lb4 = loss_g1_val_lb4, \
                             loss_g2_lb1 = loss_g2_val_lb1, loss_g2_lb2 = loss_g2_val_lb2, loss_g2_lb3 = loss_g2_val_lb3, loss_g2_lb4 = loss_g2_val_lb4, \
                             lr=optimizer.param_groups[0]['lr'])
        metric_logger.meters['samples/s'].update(batch_size / (time.time() - start_time))

        # Collect memory stats per batch
        if args.enable_profiling:
            epoch_cpu_memory.append(get_cpu_memory_usage())
            gpu_usage = get_gpu_memory_usage()
            epoch_gpu_allocated.append(gpu_usage[0])
            epoch_gpu_reserved.append(gpu_usage[1])

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
        metric_logger.meters['samples/s'].update(data_Vcomp.shape[0] / (time.time() - start_time))

        if writer:
            writer.add_scalar('loss', loss_val, step)
            writer.add_scalar('loss_g1_lb1', loss_g1_val_lb1, step)
            writer.add_scalar('loss_g1_lb2', loss_g1_val_lb2, step)
            writer.add_scalar('loss_g1_lb3', loss_g1_val_lb3, step)
            writer.add_scalar('loss_g1_lb4', loss_g1_val_lb4, step)
            writer.add_scalar('loss_g2_lb1', loss_g2_val_lb1, step)
            writer.add_scalar('loss_g2_lb2', loss_g2_val_lb2, step)
            writer.add_scalar('loss_g2_lb3', loss_g2_val_lb3, step)
            writer.add_scalar('loss_g2_lb4', loss_g2_val_lb4, step)


        step += 1
        lr_scheduler.step()

    # Calculate average memory usage
    if int(os.environ.get('SLURM_PROCID', 0)) == 0:
        if args.enable_profiling:
            avg_cpu = sum(epoch_cpu_memory) / len(epoch_cpu_memory)
            avg_gpu_allocated = sum(epoch_gpu_allocated) / len(epoch_gpu_allocated)
            avg_gpu_reserved = sum(epoch_gpu_reserved) / len(epoch_gpu_reserved)
            print(f"Epoch {epoch} Avg Memory Usage - CPU: {avg_cpu:.4f} GB | "
                f"GPU Allocated: {avg_gpu_allocated:.4f} GB | GPU Reserved: {avg_gpu_reserved:.4f} GB")
            ### calculate minimum and maximum memory usage
            min_cpu = min(epoch_cpu_memory)
            max_cpu = max(epoch_cpu_memory)
            min_gpu_allocated = min(epoch_gpu_allocated)
            max_gpu_allocated = max(epoch_gpu_allocated)
            min_gpu_reserved = min(epoch_gpu_reserved)
            max_gpu_reserved = max(epoch_gpu_reserved)
            print(f"Epoch {epoch} Memory Usage Range - CPU: {min_cpu:.4f} GB ~ {max_cpu:.4f} GB | "
                f"GPU Allocated: {min_gpu_allocated:.4f} GB ~ {max_gpu_allocated:.4f} GB | GPU Reserved: {min_gpu_reserved:.4f} GB ~ {max_gpu_reserved:.4f} GB")
            ### write memory usage to csv file
            log_memory_to_csv("train_memory_usage.csv", epoch, avg_cpu, avg_gpu_allocated, avg_gpu_reserved, min_cpu, max_cpu, min_gpu_allocated, max_gpu_allocated, min_gpu_reserved, max_gpu_reserved)


    return metric_logger.loss.global_avg, metric_logger.loss_g1_lb1.global_avg, metric_logger.loss_g1_lb2.global_avg, metric_logger.loss_g1_lb3.global_avg, metric_logger.loss_g1_lb4.global_avg, \
                                          metric_logger.loss_g2_lb1.global_avg, metric_logger.loss_g2_lb2.global_avg, metric_logger.loss_g2_lb3.global_avg, metric_logger.loss_g2_lb4.global_avg

def evaluate(model, criterion, dataloader, device, writer, args):
    global step

    # Switching model to evaluation mode
    model.eval()

    # Collect memory stats
    val_cpu_memory = []
    val_gpu_allocated = []
    val_gpu_reserved = []

    metric_logger = utils.MetricLogger(delimiter='  ')

    # This ensures no gradients are computed, saving memory and slightly speeding up computation
    header = 'Test:'

    if args.enable_profiling:
        cpu_usage_val_start = get_cpu_memory_usage()
        gpu_usage_val_start = get_gpu_memory_usage()
        print(f"Validation Start - CPU: {cpu_usage_val_start:.4f} GB | "
              f"GPU: {gpu_usage_val_start[0]:.4f}/{gpu_usage_val_start[1]:.4f} GB")

    with torch.no_grad():
        for data_Vcomp, data_Hcomp, label1, label2, label3, label4 in metric_logger.log_every(dataloader, 20, header):
            data_Vcomp, data_Hcomp = data_Vcomp.to(device, non_blocking=True), data_Hcomp.to(device, non_blocking=True)
            label1 = label1.to(device, non_blocking=True)
            label2 = label2.to(device, non_blocking=True)
            label3 = label3.to(device, non_blocking=True)
            label4 = label4.to(device, non_blocking=True)

            output_lb1, output_lb2, output_lb3, output_lb4  = model(data_Vcomp, data_Hcomp)

            loss, loss_g1_lb1, loss_g1_lb2, loss_g1_lb3, loss_g1_lb4, \
                  loss_g2_lb1, loss_g2_lb2, loss_g2_lb3, loss_g2_lb4 = criterion(output_lb1, output_lb2, output_lb3, output_lb4, \
                                                                                label1, label2, label3, label4)
            if args.enable_profiling:
                val_cpu_memory.append(get_cpu_memory_usage())
                gpu_usage = get_gpu_memory_usage()
                val_gpu_allocated.append(gpu_usage[0])
                val_gpu_reserved.append(gpu_usage[1])

            metric_logger.update(loss=loss.item(), loss_g1_lb1=loss_g1_lb1.item(), loss_g1_lb2=loss_g1_lb2.item(), loss_g1_lb3=loss_g1_lb3.item(), loss_g1_lb4=loss_g1_lb4.item(), \
                                                   loss_g2_lb1=loss_g2_lb1.item(), loss_g2_lb2=loss_g2_lb2.item(), loss_g2_lb3=loss_g2_lb3.item(), loss_g2_lb4=loss_g2_lb4.item())

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(' * Loss {loss.global_avg:.8f}\n'.format(loss=metric_logger.loss))
    if writer:
        writer.add_scalar('loss', metric_logger.loss.global_avg, step)
        writer.add_scalar('loss_g1_lb1', metric_logger.loss_g1_lb1.global_avg, step)
        writer.add_scalar('loss_g1_lb2', metric_logger.loss_g1_lb2.global_avg, step)
        writer.add_scalar('loss_g1_lb3', metric_logger.loss_g1_lb3.global_avg, step)
        writer.add_scalar('loss_g1_lb4', metric_logger.loss_g1_lb4.global_avg, step)
        writer.add_scalar('loss_g2_lb1', metric_logger.loss_g2_lb1.global_avg, step)
        writer.add_scalar('loss_g2_lb2', metric_logger.loss_g2_lb2.global_avg, step)
        writer.add_scalar('loss_g2_lb3', metric_logger.loss_g2_lb3.global_avg, step)
        writer.add_scalar('loss_g2_lb4', metric_logger.loss_g2_lb4.global_avg, step)

    if int(os.environ.get('SLURM_PROCID', 0)) == 0:
        if args.enable_profiling:
            avg_cpu = sum(val_cpu_memory) / len(val_cpu_memory)
            avg_gpu_allocated = sum(val_gpu_allocated) / len(val_gpu_allocated)
            avg_gpu_reserved = sum(val_gpu_reserved) / len(val_gpu_reserved)
            print(f"Validation Avg Memory Usage - CPU: {avg_cpu:.4f} GB | "
                f"GPU Allocated: {avg_gpu_allocated:.4f} GB | GPU Reserved: {avg_gpu_reserved:.4f} GB")
            ### calculate minimum and maximum memory usage
            min_cpu = min(val_cpu_memory)
            max_cpu = max(val_cpu_memory)
            min_gpu_allocated = min(val_gpu_allocated)
            max_gpu_allocated = max(val_gpu_allocated)
            min_gpu_reserved = min(val_gpu_reserved)
            max_gpu_reserved = max(val_gpu_reserved)
            print(f"Validation Memory Usage - CPU: min: {min_cpu:.4f} GB, max: {max_cpu:.4f} GB | "
                f"GPU Allocated: min: {min_gpu_allocated:.4f} GB, max: {max_gpu_allocated:.4f} GB")
            log_memory_to_csv("val_memory_usage.csv", args.start_epoch, avg_cpu, avg_gpu_allocated, avg_gpu_reserved, min_cpu, max_cpu, min_gpu_allocated, max_gpu_allocated, min_gpu_reserved, max_gpu_reserved)

    return metric_logger.loss.global_avg, metric_logger.loss_g1_lb1.global_avg, metric_logger.loss_g1_lb2.global_avg, metric_logger.loss_g1_lb3.global_avg, metric_logger.loss_g1_lb4.global_avg, \
                                          metric_logger.loss_g2_lb1.global_avg, metric_logger.loss_g2_lb2.global_avg, metric_logger.loss_g2_lb3.global_avg, metric_logger.loss_g2_lb4.global_avg
def main(args):
    global step

    print(args)
    # Outputting torch versions
    print('torch version: ', torch.__version__)
    print('torchvision version: ', torchvision.__version__)

    # If the output path specified does not exist, make the directory
    utils.mkdir(args.output_path)  # create folder to store checkpoints
    utils.init_distributed_mode(args)  # distributed mode initialization

    # Set up tensorboard summary writer
    train_writer, val_writer = None, None
    if args.tensorboard:
        utils.mkdir(args.log_path)  # create folder to store tensorboard logs
        if not args.distributed or (args.rank == 0) and (args.local_rank == 0):
            train_writer = SummaryWriter(os.path.join(args.output_path, 'logs', 'train'))
            val_writer = SummaryWriter(os.path.join(args.output_path, 'logs', 'val'))

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    with open('dataset_config.json') as f:
        try:
            ctx = json.load(f)[args.dataset]
        except KeyError:
            print('Unsupported dataset.')
            sys.exit()

    if args.file_size is not None:
        ctx['file_size'] = args.file_size

    # Create dataset and dataloader
    print('Loading data')
    print('Loading training data')

    # Normalize data and label to [-1, 1]
    transform_data1 = Compose([
        t.LogTransform(k=args.k),
        t.MinMaxNormalize(t.log_transform(ctx['data_y_min'], k=args.k), t.log_transform(ctx['data_y_max'], k=args.k))
    ])
    transform_data2 = Compose([
        t.LogTransform(k=args.k),
        t.MinMaxNormalize(t.log_transform(ctx['data_x_min'], k=args.k), t.log_transform(ctx['data_x_max'], k=args.k))
    ])
    transform_label1 = Compose([
        t.MinMaxNormalize(ctx['label1_min'], ctx['label1_max'])
    ])
    transform_label2 = Compose([
        t.MinMaxNormalize(ctx['label2_min'], ctx['label2_max'])
    ])
    transform_data3 = Compose([
        t.MinMaxNormalize(ctx['label3_min'], ctx['label3_max'])
    ])
    transform_data4 = Compose([
        t.MinMaxNormalize(ctx['label4_min'], ctx['label4_max'])
    ])


    print("train_anno is txt", args.train_anno)
    print("sample_ratio is ", args.sample_temporal)
    print("file_size is ", ctx['file_size'])
    if args.train_anno[-3:] == 'txt':
        dataset_train = FWIDataset(
            args.train_anno,
            preload=True,
            sample_ratio=args.sample_temporal,
            file_size=ctx['file_size'],
            transform_data1=transform_data1,
            transform_data2=transform_data2,
            transform_label1=transform_label1,
            transform_label2=transform_label2,
            transform_label3=transform_data3,
            transform_label4=transform_data4
        )
    else:
        dataset_train = torch.load(args.train_anno)

    print('Loading validation data')
    if args.val_anno[-3:] == 'txt':
        dataset_valid = FWIDataset(
            args.val_anno,
            preload=True,
            sample_ratio=args.sample_temporal,
            file_size=ctx['file_size'],
            transform_data1=transform_data1,
            transform_data2=transform_data2,
            transform_label1=transform_label1,
            transform_label2=transform_label2,
            transform_label3=transform_data3,
            transform_label4=transform_data4)
    else:
        dataset_valid = torch.load(args.val_anno)

    print('Creating data loaders')
    if args.distributed:
        train_sampler = DistributedSampler(dataset_train, shuffle=True)
        valid_sampler = DistributedSampler(dataset_valid, shuffle=True)
    else:
        train_sampler = RandomSampler(dataset_train)
        valid_sampler = RandomSampler(dataset_valid)

    dataloader_train = DataLoader(
        dataset_train, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers,
        pin_memory=True, drop_last=True, collate_fn=default_collate)

    dataloader_valid = DataLoader(
        dataset_valid, batch_size=args.batch_size,
        sampler=valid_sampler, num_workers=args.workers,
        pin_memory=True, collate_fn=default_collate)

    print('Creating model')
    if args.model not in network.model_dict:
        print('Unsupported model.')
        sys.exit()
    
    # Freeze all the layers of encoders and decoders
    def freeze_encoders_decoders(model):
        for name, param in model.named_parameters():
            # Freeze all encoder layers
            if 'encoder_with_lstm' in name:
                param.requires_grad = False
            # Freeze all decoder layers
            if 'decoders' in name:
                param.requires_grad = False

        for name, param in model.named_parameters():
            # if any(layer in name for layer in ['deconv6', 'deconv3_1', 'deconv3_2', 'convblock8']):
            if any(layer in name for layer in ['deconv6', 'deconv3_1', 'deconv3_2', 'convblock8']):
                param.requires_grad = True

    model = network.model_dict[args.model](upsample_mode=args.up_mode, sample_spatial=args.sample_spatial,
                                           sample_temporal=args.sample_temporal).to(device)

    # Call this function after initializing the model
    # freeze_encoders_decoders(model)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Define loss function (Updated loss function to handle two outputs (velocity and saturation)
    l1loss = nn.L1Loss()
    l2loss = nn.MSELoss()

    def criterion(pred_lb1, pred_lb2, pred_lb3, pred_lb4, gt_lb1, gt_lb2, gt_lb3, gt_lb4):
        loss_g1_lb1 = l1loss(pred_lb1, gt_lb1)
        loss_g1_lb2 = l1loss(pred_lb2, gt_lb2)
        loss_g1_lb3 = l1loss(pred_lb3, gt_lb3)
        loss_g1_lb4 = l1loss(pred_lb4, gt_lb4)
        loss_g2_lb1 = l2loss(pred_lb1, gt_lb1)
        loss_g2_lb2 = l2loss(pred_lb2, gt_lb2)
        loss_g2_lb3 = l2loss(pred_lb3, gt_lb3)
        loss_g2_lb4 = l2loss(pred_lb4, gt_lb4)
        
        loss = (args.lambda_g1_lb1 * loss_g1_lb1) + (args.lambda_g1_lb2 * loss_g1_lb2) + (args.lambda_g1_lb3 * loss_g1_lb3) + (args.lambda_g1_lb4 * loss_g1_lb4) + \
               (args.lambda_g2_lb1 * loss_g2_lb1) + (args.lambda_g2_lb2 * loss_g2_lb2) + (args.lambda_g2_lb3 * loss_g2_lb3) + (args.lambda_g2_lb4 * loss_g2_lb4)
                
        return loss, loss_g1_lb1, loss_g1_lb2, loss_g1_lb3, loss_g1_lb4, \
                     loss_g2_lb1, loss_g2_lb2, loss_g2_lb3, loss_g2_lb4
    

    # Scale lr according to effective batch size
    lr = args.lr * args.world_size
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    # optimizer = torch.optim.AdamW([param for param in model.parameters() if param.requires_grad], lr=lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    # Convert scheduler to be per iteration instead of per epoch
    warmup_iters = args.lr_warmup_epochs * len(dataloader_train)
    lr_milestones = [len(dataloader_train) * m for m in args.lr_milestones]
    lr_scheduler = WarmupMultiStepLR(
        optimizer, milestones=lr_milestones, gamma=args.lr_gamma,
        warmup_iters=warmup_iters, warmup_factor=1e-5)

    model_without_ddp = model
    if args.distributed:
        # model = DistributedDataParallel(model, device_ids=[args.local_rank])
        model = DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(network.replace_legacy(checkpoint['model']))
        optimizer.load_state_dict(checkpoint['optimizer'])
        # # Override the learning rate from the checkpoint with the command-line value:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * args.world_size
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        
        # Set the starting epoch and step
        # args.start_epoch = checkpoint['epoch'] + 1
        # step = checkpoint['step']
        # lr_scheduler.milestones = lr_milestones

        # Step 3: Reset the epoch count to 0
        args.start_epoch = 0
        step = 0  # You might also want to reset the step count if applicable
        lr_scheduler.milestones = lr_milestones


    print('Start training')
    start_time = time.time()
    best_loss = 10
    chp = 1

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # train_one_epoch(model, criterion, optimizer, lr_scheduler, dataloader_train,
        #                 device, epoch, args.print_freq, train_writer, args)

        # Collect training losses
        train_loss, train_loss_g1_lb1, train_loss_g1_lb2, train_loss_g1_lb3, train_loss_g1_lb4, \
                    train_loss_g2_lb1, train_loss_g2_lb2, train_loss_g2_lb3, train_loss_g2_lb4 = \
                                                            train_one_epoch(model, criterion, optimizer, lr_scheduler, dataloader_train, device, epoch, args.print_freq, train_writer, args)

        loss, loss_g1_lb1, loss_g1_lb2, loss_g1_lb3, loss_g1_lb4, \
              loss_g2_lb1, loss_g2_lb2, loss_g2_lb3, loss_g2_lb4 = \
                                                            evaluate(model, criterion, dataloader_valid, device, val_writer, args)

        checkpoint = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'step': step,
            'args': args}
        # Save checkpoint per epoch
        if loss < best_loss:
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_path, 'checkpoint.pth'))
            print('saving checkpoint at epoch: ', epoch)
            chp = epoch
            best_loss = loss
        # Save checkpoint every epoch block
        print('current best loss: ', best_loss)
        print('current best epoch: ', chp)
        if args.output_path and (epoch + 1) % args.epoch_block == 0:
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_path, 'model_{}.pth'.format(epoch + 1)))

        if int(os.environ.get('SLURM_PROCID', 0)) == 0:
            update_csv("loss_func.csv", epoch, loss, loss_g1_lb1, loss_g1_lb2, loss_g1_lb3, loss_g1_lb4, \
                                                     loss_g2_lb1, loss_g2_lb2, loss_g2_lb3, loss_g2_lb4, \
                        train_loss, train_loss_g1_lb1, train_loss_g1_lb2, train_loss_g1_lb3, train_loss_g1_lb4, \
                                    train_loss_g2_lb1, train_loss_g2_lb2, train_loss_g2_lb3, train_loss_g2_lb4)
                       
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='FCN Training')
    parser.add_argument('-d', '--device', default='cuda', help='device')
    parser.add_argument('-ds', '--dataset', default='flatfault-b', type=str, help='dataset name')
    parser.add_argument('-fs', '--file-size', default=None, type=int, help='number of samples in each npy file')
    parser.add_argument('--accumulation-steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--enable_profiling', action='store_true', help='Enable memory profiling during training and validation.')

    # Path related
    parser.add_argument('-ap', '--anno-path', default='split_files', help='annotation files location')
    parser.add_argument('-t', '--train-anno', default='flatfault_b_train_invnet.txt', help='name of train anno')
    parser.add_argument('-v', '--val-anno', default='flatfault_b_val_invnet.txt', help='name of val anno')
    parser.add_argument('-o', '--output-path', default='Invnet_models', help='path to parent folder to save checkpoints')
    parser.add_argument('-l', '--log-path', default='Invnet_models', help='path to parent folder to save logs')
    parser.add_argument('-n', '--save-name', default='fcn_l1loss_ffb', help='folder name for this experiment')
    parser.add_argument('-s', '--suffix', type=str, default=None, help='subfolder name for this run')

    # Model related
    parser.add_argument('-m', '--model', type=str, help='inverse model name')
    parser.add_argument('-um', '--up-mode', default=None,
                        help='upsampling layer mode such as "nearest", "bicubic", etc.')
    parser.add_argument('-ss', '--sample-spatial', type=float, default=1.0, help='spatial sampling ratio')
    parser.add_argument('-st', '--sample-temporal', type=int, default=1, help='temporal sampling ratio')
    # Training related
    parser.add_argument('-b', '--batch-size', default=256, type=int)
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('-lm', '--lr-milestones', nargs='+', default=[], type=int, help='decrease lr on milestones')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lr-warmup-epochs', default=0, type=int, help='number of warmup epochs')
    parser.add_argument('-eb', '--epoch_block', type=int, default=40, help='epochs in a saved block')
    parser.add_argument('-nb', '--num_block', type=int, default=3, help='number of saved block')
    parser.add_argument('-j', '--workers', default=6, type=int, help='number of data loading workers (default: 16)')
    parser.add_argument('--k', default=1, type=float, help='k in log transformation')
    parser.add_argument('--print-freq', default=50, type=int, help='print frequency')
    parser.add_argument('-r', '--resume', default=None, help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, help='start epoch')

    # Loss related
    parser.add_argument('-g1_lb1', '--lambda_g1_lb1', type=float, default=1.0)
    parser.add_argument('-g1_lb2', '--lambda_g1_lb2', type=float, default=1.0)
    parser.add_argument('-g1_lb3', '--lambda_g1_lb3', type=float, default=1.0)
    parser.add_argument('-g1_lb4', '--lambda_g1_lb4', type=float, default=1.0)
    parser.add_argument('-g2_lb1', '--lambda_g2_lb1', type=float, default=1.0)
    parser.add_argument('-g2_lb2', '--lambda_g2_lb2', type=float, default=1.0)
    parser.add_argument('-g2_lb3', '--lambda_g2_lb3', type=float, default=1.0)
    parser.add_argument('-g2_lb4', '--lambda_g2_lb4', type=float, default=1.0)

    # Distributed training related
    parser.add_argument('--sync-bn', action='store_true', help='Use sync batch norm')
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    # Tensorboard related
    parser.add_argument('--tensorboard', action='store_true', help='Use tensorboard for logging.')

    args = parser.parse_args()

    args.output_path = os.path.join(args.output_path, args.save_name, args.suffix or '')
    args.log_path = os.path.join(args.log_path, args.save_name, args.suffix or '')
    args.train_anno = os.path.join(args.anno_path, args.train_anno)
    args.val_anno = os.path.join(args.anno_path, args.val_anno)

    args.epochs = args.epoch_block * args.num_block

    if args.resume:
        args.resume = os.path.join(args.output_path, args.resume)

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
