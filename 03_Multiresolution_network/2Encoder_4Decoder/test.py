import sys
import time
import datetime
import json

import torch.nn as nn
import torchvision
from torch.utils.data import SequentialSampler
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import Compose

import utils
import network
from vis import *
from dataset import FWIDataset
import transforms as t
import pytorch_ssim


def process_label(label_name, label, ctx):
    """
    Process label by denormalizing it.

    :param label: input label
    :param ctx: context containing minimum and maximum values for denormalization
    :return: a tuple of denormalized numpy array and original label
    """
    label_np = t.tonumpy_denormalize(label, ctx[f'{label_name}_min'], ctx[f'{label_name}_max'], exp=False)

    return label_np, label

def process_data1(data, ctx, k):
    """
    Process data by denormalizing it.

    :param data: input data
    :param ctx: context containing minimum and maximum values for denormalization
    :param k: integer for denormalization
    :return: denormalized data
    """
    return t.tonumpy_denormalize(data, ctx['data_y_min'], ctx['data_y_max'], k=k)

def process_data2(data, ctx, k):
    """
    Process data by denormalizing it.

    :param data: input data
    :param ctx: context containing minimum and maximum values for denormalization
    :param k: integer for denormalization
    :return: denormalized data
    """
    return t.tonumpy_denormalize(data, ctx['data_x_min'], ctx['data_x_max'], k=k)

def process_label_and_append(label_name, label, ctx, label_list, label_tensor):
    """
    Process label and append it to the provided lists.

    :param label: input label
    :param ctx: context for label processing
    :param label_list: list to append the processed label in numpy format
    :param label_tensor: list to append the processed label in tensor format
    :return: a tuple of processed label in numpy and original format
    """
    label_np, label = process_label(label_name, label, ctx)
    label_list.append(label_np)
    label_tensor.append(label)
    return label_np, label

def calculate_and_print_losses(name, label, pred):
    """
    Calculate and print L1, MSE and SSIM losses.

    :param name: name of the context (e.g., 'velocity', 'saturation')
    :param label: ground truth
    :param pred: model's prediction
    :return: None
    """
    l1 = nn.L1Loss()
    l2 = nn.MSELoss()
    ssim_loss = pytorch_ssim.SSIM(window_size=11)

    mae = l1(label, pred)
    mse = l2(label, pred)
    ssim = ssim_loss(label / 2 + 0.5, pred / 2 + 0.5)  # (-1, 1) to (0, 1)

    print(f'MAE ({name}): {mae}')
    print(f'MSE ({name}): {mse}')
    print(f'SSIM ({name}): {ssim}')


def evaluate(model, criterions, dataloader, device, k, ctx, vis_path, vis_batch, vis_sample, missing, std) -> None:
    """
    Evaluate the model and calculate losses.
    For the purpose of the brevity and the complexity of this function, a partial docstring is provided.
    You should complete the docstring according to your need.

    :param model: a model to be evaluated
    :param criterions: a dictionary with loss functions
    :param dataloader: a dataloader with data for evaluation
    :param device: device to run the model
    :param k: integer for denormalization
    :param ctx: context containing minimum and maximum values for denormalization
    :param vis_path: path for visualization
    :param vis_batch: number of visualization batches
    :param vis_sample: number of visualization samples
    :param missing: to show whether there is any missing data
    :param std: standard deviation for adding gaussian noise
    :return: None
    """
    model.eval()

    # store denormalized prediction & gt in numpy
    label1_pred_list, label2_pred_list, label3_pred_list, label4_pred_list = [], [], [], []
        
    label1_list, label2_list, label3_list, label4_list = [], [], [], []

    # store normalized prediction & gt in tensor
    label1_tensor, label2_tensor, label3_tensor, label4_tensor = [], [], [], []
    
    label1_pred_tensor, label2_pred_tensor, label3_pred_tensor, label4_pred_tensor = [], [], [], []

    if missing or std:
        data_y_list, data_x_list, data_y_noise_list, data_x_noise_list = [], [], [], []  # store original data and noisy/muted data

    with torch.no_grad():
        batch_idx = 0
        for data_y, data_x, label1, label2, label3, label4 in dataloader:
            data_y = data_y.type(torch.FloatTensor).to(device, non_blocking=True)
            data_x = data_x.type(torch.FloatTensor).to(device, non_blocking=True)
            
            label1 = label1.type(torch.FloatTensor).to(device, non_blocking=True)
            label2 = label2.type(torch.FloatTensor).to(device, non_blocking=True)
            label3 = label3.type(torch.FloatTensor).to(device, non_blocking=True)
            label4 = label4.type(torch.FloatTensor).to(device, non_blocking=True)

            label1_np, label1 = process_label_and_append('label1', label1, ctx, label1_list, label1_tensor)
            label2_np, label2 = process_label_and_append('label2', label2, ctx, label2_list, label2_tensor)
            label3_np, label3 = process_label_and_append('label3', label3, ctx, label3_list, label3_tensor)
            label4_np, label4 = process_label_and_append('label4', label4, ctx, label4_list, label4_tensor)

            if missing or std:
                # Add gaussian noise
                data_y_noise = torch.clip(
                    data_y + (std ** 0.5) * torch.randn(data_y.shape).to(device, non_blocking=True), min=-1, max=1)
                data_x_noise = torch.clip(
                    data_x + (std ** 0.5) * torch.randn(data_x.shape).to(device, non_blocking=True), min=-1, max=1)

                # Mute some traces
                mute_idx1 = np.random.choice(data_y.shape[3], size=missing, replace=False)
                data_y_noise[:, :, :, mute_idx1] = data_y[0, 0, 0, 0]

                mute_idx2 = np.random.choice(data_x.shape[3], size=missing, replace=False)
                data_x_noise[:, :, :, mute_idx2] = data_x[0, 0, 0, 0]

                data_y_np = t.tonumpy_denormalize(data_y, ctx['data_y_min'], ctx['data_y_max'], k=k)
                data_y_noise_np = t.tonumpy_denormalize(data_y_noise, ctx['data_y_min'], ctx['data_y_max'], k=k)
                data_y_list.append(data_y_np)
                data_y_noise_list.append(data_y_noise_np)

                data_x_np = t.tonumpy_denormalize(data_x, ctx['data_x_min'], ctx['data_x_max'], k=k)
                data_x_noise_np = t.tonumpy_denormalize(data_x_noise, ctx['data_x_min'], ctx['data_x_max'], k=k)
                data_x_list.append(data_x_np)
                data_x_noise_list.append(data_x_noise_np)

                pred_lb1, pred_lb2, pred_lb3, pred_lb4 = model(data_y_noise, data_x_noise)
            else:
                pred_lb1, pred_lb2, pred_lb3, pred_lb4 = model(data_y, data_x)

            label1_pred_np = t.tonumpy_denormalize(pred_lb1, ctx['label1_min'], ctx['label1_max'], exp=False)
            label1_pred_list.append(label1_pred_np)
            label1_pred_tensor.append(pred_lb1)

            label2_pred_np = t.tonumpy_denormalize(pred_lb2, ctx['label2_min'], ctx['label2_max'], exp=False)
            label2_pred_list.append(label2_pred_np)
            label2_pred_tensor.append(pred_lb2)

            label3_pred_np = t.tonumpy_denormalize(pred_lb3, ctx['label3_min'], ctx['label3_max'], exp=False)
            label3_pred_list.append(label3_pred_np)
            label3_pred_tensor.append(pred_lb3)

            label4_pred_np = t.tonumpy_denormalize(pred_lb4, ctx['label4_min'], ctx['label4_max'], exp=False)
            label4_pred_list.append(label4_pred_np)
            label4_pred_tensor.append(pred_lb4)

            # Visualization
            if vis_path and batch_idx < vis_batch:
                # Determine the number of samples available for visualization.
                num_vis = min(vis_sample, label1_pred_np.shape[1], label1_np.shape[0])
                for i in range(num_vis):
                    plot_velocity(label1_pred_np[0, i, 0].squeeze(), label1_np[i, 0].squeeze(), f'{vis_path}/L1_{batch_idx}_{i}.png')
                    plot_velocity(label2_pred_np[0, i, 0].squeeze(), label2_np[i, 0].squeeze(), f'{vis_path}/L2_{batch_idx}_{i}.png')
                    plot_velocity(label3_pred_np[0, i, 0].squeeze(), label3_np[i, 0].squeeze(), f'{vis_path}/L3_{batch_idx}_{i}.png')
                    plot_velocity(label4_pred_np[0, i, 0].squeeze(), label4_np[i, 0].squeeze(), f'{vis_path}/L4_{batch_idx}_{i}.png')
                    #### save predicted velocity as npy
                    np.save(f'{vis_path}/L1_{batch_idx}_{i}.npy', label1_pred_np[i, 0])
                    np.save(f'{vis_path}/L2_{batch_idx}_{i}.npy', label2_pred_np[i, 0])
                    np.save(f'{vis_path}/L3_{batch_idx}_{i}.npy', label3_pred_np[i, 0])
                    np.save(f'{vis_path}/L4_{batch_idx}_{i}.npy', label4_pred_np[i, 0])
                    #### save ground truth as npy
                    np.save(f'{vis_path}/L1_{batch_idx}_{i}_gt.npy', label1_np[i, 0])
                    np.save(f'{vis_path}/L2_{batch_idx}_{i}_gt.npy', label2_np[i, 0])
                    np.save(f'{vis_path}/L3_{batch_idx}_{i}_gt.npy', label3_np[i, 0])
                    np.save(f'{vis_path}/L4_{batch_idx}_{i}_gt.npy', label4_np[i, 0])

                    if missing or std:
                        for ch in [2]:  # range(data.shape[1]):
                            plot_seismic(data_y_np[i, ch], data_y_noise_np[i, ch],
                                         f'{vis_path}/Svp_{batch_idx}_{i}_{ch}.png', vmin=ctx['data1_min'] * 0.01,
                                         vmax=ctx['data1_max'] * 0.01)
                            plot_seismic(data_x_np[i, ch], data_x_noise_np[i, ch],
                                         f'{vis_path}/Svs_{batch_idx}_{i}_{ch}.png', vmin=ctx['data2_min'] * 0.01,
                                         vmax=ctx['data2_max'] * 0.01)
            batch_idx += 1

    label1, label1_pred = np.concatenate(label1_list), np.concatenate(label1_pred_list)
    label2, label2_pred = np.concatenate(label2_list), np.concatenate(label2_pred_list)
    label3, label3_pred = np.concatenate(label3_list), np.concatenate(label3_pred_list)
    label4, label4_pred = np.concatenate(label4_list), np.concatenate(label4_pred_list)

    label1_t, pred_lb1_t = torch.cat(label1_tensor), torch.cat(label1_pred_tensor)
    label2_t, pred_lb2_t = torch.cat(label2_tensor), torch.cat(label2_pred_tensor)
    label3_t, pred_lb3_t = torch.cat(label3_tensor), torch.cat(label3_pred_tensor)
    label4_t, pred_lb4_t = torch.cat(label4_tensor), torch.cat(label4_pred_tensor)

    # Debug prints:
    print("Before flattening:")
    print("Model (velocity - L1) output size:", pred_lb1_t.size())
    print("Model (velocity - L2) output size:", pred_lb2_t.size())
    print("Model (velocity - L3) output size:", pred_lb3_t.size())
    print("Model (velocity - L4) output size:", pred_lb4_t.size())
    print("Target (velocity - L1) size:", label1_t.size())
    print("Target (velocity - L2) size:", label2_t.size())
    print("Target (velocity - L3) size:", label3_t.size())
    print("Target (velocity - L4) size:", label4_t.size())
    # Flatten the batch and time dimensions only if the tensor is 5D
    if pred_lb1_t.dim() == 5:
        pred_lb1_t = pred_lb1_t.view(-1, pred_lb1_t.size(2), pred_lb1_t.size(3), pred_lb1_t.size(4))
    if label1_t.dim() == 5:
        label1_t   = label1_t.view(-1, label1_t.size(2), label1_t.size(3), label1_t.size(4))
    if pred_lb2_t.dim() == 5:
        pred_lb2_t = pred_lb2_t.view(-1, pred_lb2_t.size(2), pred_lb2_t.size(3), pred_lb2_t.size(4))
    if label2_t.dim() == 5:
        label2_t   = label2_t.view(-1, label2_t.size(2), label2_t.size(3), label2_t.size(4))
    if pred_lb3_t.dim() == 5:
        pred_lb3_t = pred_lb3_t.view(-1, pred_lb3_t.size(2), pred_lb3_t.size(3), pred_lb3_t.size(4))
    if label3_t.dim() == 5:
        label3_t   = label3_t.view(-1, label3_t.size(2), label3_t.size(3), label3_t.size(4))
    if pred_lb4_t.dim() == 5:
        pred_lb4_t = pred_lb4_t.view(-1, pred_lb4_t.size(2), pred_lb4_t.size(3), pred_lb4_t.size(4))
    if label4_t.dim() == 5:
        label4_t   = label4_t.view(-1, label4_t.size(2), label4_t.size(3), label4_t.size(4))
    print("After flattening:")
    print("Model (velocity - L1) output size:", pred_lb1_t.size())
    print("Model (velocity - L2) output size:", pred_lb2_t.size())
    print("Model (velocity - L3) output size:", pred_lb3_t.size())
    print("Model (velocity - L4) output size:", pred_lb4_t.size())
    print("Target (velocity - L1) size:", label1_t.size())
    print("Target (velocity - L2) size:", label2_t.size())
    print("Target (velocity - L3) size:", label3_t.size())
    print("Target (velocity - L4) size:", label4_t.size())

    calculate_and_print_losses('velocity_l1', label1_t, pred_lb1_t)
    calculate_and_print_losses('velocity_l2', label2_t, pred_lb2_t)
    calculate_and_print_losses('velocity_l3', label3_t, pred_lb3_t)
    calculate_and_print_losses('velocity_l4', label4_t, pred_lb4_t)

    for name, criterion in criterions.items():
        print(f' * Velocity {name}: {criterion(label1, label1_pred)}')
        print(f' * Velocity {name}: {criterion(label2, label2_pred)}')
        print(f' * Velocity {name}: {criterion(label3, label3_pred)}')
        print(f' * Velocity {name}: {criterion(label4, label4_pred)}')
        

def main(args):
    print(args)
    print("torch version: ", torch.__version__)
    print("torchvision version: ", torchvision.__version__)

    utils.mkdir(args.output_path)
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

    print("Loading data")
    print("Loading validation data")
    log_data_y_min = t.log_transform(ctx['data_y_min'], k=args.k)
    log_data_y_max = t.log_transform(ctx['data_y_max'], k=args.k)

    log_data_x_min = t.log_transform(ctx['data_x_min'], k=args.k)
    log_data_x_max = t.log_transform(ctx['data_x_max'], k=args.k)

    transform_valid_data1 = Compose([t.LogTransform(k=args.k), t.MinMaxNormalize(log_data_y_min, log_data_y_max), ])
    transform_valid_data2 = Compose([t.LogTransform(k=args.k), t.MinMaxNormalize(log_data_x_min, log_data_x_max), ])
    transform_valid_label1 = Compose([t.MinMaxNormalize(ctx['label1_min'], ctx['label1_max'])])
    transform_valid_label2 = Compose([t.MinMaxNormalize(ctx['label2_min'], ctx['label2_max'])])
    transform_valid_label3 = Compose([t.MinMaxNormalize(ctx['label3_min'], ctx['label3_max'])])
    transform_valid_label4 = Compose([t.MinMaxNormalize(ctx['label4_min'], ctx['label4_max'])])

    if args.val_anno[-3:] == 'txt':
        dataset_valid = FWIDataset(
            args.val_anno,
            sample_ratio=args.sample_temporal,
            file_size=ctx['file_size'],
            transform_data1=transform_valid_data1,
            transform_data2=transform_valid_data2,
            transform_label1=transform_valid_label1,
            transform_label2=transform_valid_label2,
            transform_label3=transform_valid_label3,
            transform_label4=transform_valid_label4
        )
    else:
        dataset_valid = torch.load(args.val_anno)

    print("Creating data loaders")
    valid_sampler = SequentialSampler(dataset_valid)
    dataloader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=args.batch_size,
        sampler=valid_sampler, num_workers=args.workers,
        pin_memory=True, collate_fn=default_collate)

    print("Creating model")
    if args.model not in network.model_dict:
        print('Unsupported model.')
        sys.exit()

    model = network.model_dict[args.model](upsample_mode=args.up_mode, sample_spatial=args.sample_spatial,
                                           sample_temporal=args.sample_temporal, norm=args.norm).to(device)

    criterions = {
        'MAE': lambda x, y: np.mean(np.abs(x - y)),
        'MSE': lambda x, y: np.mean((x - y) ** 2)
    }

    if args.resume:
        print(args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(network.replace_legacy(checkpoint['model']))
        print('Loaded model checkpoint at Epoch {} / Step {}.'.format(checkpoint['epoch'], checkpoint['step']))

    if args.vis:
        # Create folder to store visualization results
        vis_folder = f'visualization_{args.vis_suffix}' if args.vis_suffix else 'visualization'
        vis_path = os.path.join(args.output_path, vis_folder)
        utils.mkdir(vis_path)
    else:
        vis_path = None

    print("Start testing")
    start_time = time.time()
    evaluate(model, criterions, dataloader_valid, device, args.k, ctx, vis_path, args.vis_batch, args.vis_sample,
             args.missing, args.std)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Testing time {}'.format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='FCN Testing')
    parser.add_argument('-d', '--device', default='cuda', help='device')
    parser.add_argument('-ds', '--dataset', default='flatfault-b', type=str, help='dataset name')
    parser.add_argument('-fs', '--file-size', default=None, type=int, help='number of samples in each npy file')

    # Path related
    parser.add_argument('-ap', '--anno-path', default='split_files', help='annotation files location')
    parser.add_argument('-v', '--val-anno', default='flatfault_b_val_invnet.txt', help='name of val anno')
    parser.add_argument('-o', '--output-path', default='Invnet_models', help='path to parent folder to save checkpoints')
    parser.add_argument('-n', '--save-name', default='fcn_l1loss_ffb', help='folder name for this experiment')
    parser.add_argument('-s', '--suffix', type=str, default=None, help='subfolder name for this run')

    # Model related
    parser.add_argument('-m', '--model', type=str, help='inverse model name')
    parser.add_argument('-no', '--norm', default='bn', help='normalization layer type, support bn, in, ln (default: bn)')
    parser.add_argument('-um', '--up-mode', default=None, help='upsampling layer mode such as "nearest", "bicubic", etc.')
    parser.add_argument('-ss', '--sample-spatial', type=float, default=1.0, help='spatial sampling ratio')
    parser.add_argument('-st', '--sample-temporal', type=int, default=1, help='temporal sampling ratio')

    # Test related
    parser.add_argument('-b', '--batch-size', default=50, type=int)
    parser.add_argument('-j', '--workers', default=16, type=int, help='number of data loading workers (default: 16)')
    parser.add_argument('--k', default=1, type=float, help='k in log transformation')
    parser.add_argument('-r', '--resume', default=None, help='resume from checkpoint')
    parser.add_argument('--vis', help='visualization option', action="store_true")
    parser.add_argument('-vsu', '--vis-suffix', default=None, type=str, help='visualization suffix')
    parser.add_argument('-vb', '--vis-batch', help='number of batch to be visualized', default=0, type=int)
    parser.add_argument('-vsa', '--vis-sample', help='number of samples in a batch to be visualized', default=0, type=int)
    parser.add_argument('--missing', default=0, type=int, help='number of missing traces')
    parser.add_argument('--std', default=0, type=float, help='standard deviation of gaussian noise')
    parser.add_argument('--saturation', help='enable saturation augmentation', action="store_true")
    parser.add_argument('--lambda-vel', default=1.0, type=float, help='weight for velocity loss')
    args = parser.parse_args()

    args.output_path = os.path.join(args.output_path, args.save_name, args.suffix or '')
    args.val_anno = os.path.join(args.anno_path, args.val_anno)
    args.resume = os.path.join(args.output_path, args.resume)

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
