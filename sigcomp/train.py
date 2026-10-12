import os
import sys
import random
import numpy as np
import torch
import torch.ao.quantization as quant
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from .schedule import linear_warmup_cosine_decay
from .dataset import SignalDataset
from .model import QAT_MLP_Compression


def get_qat_config():
    activation_qconfig = quant.FakeQuantize.with_args(
        observer=quant.MovingAverageMinMaxObserver,
        quant_min=-128, quant_max=127,
        dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, reduce_range=False
    )
    weight_qconfig = quant.FakeQuantize.with_args(
        observer=quant.MovingAverageMinMaxObserver,
        quant_min=-127, quant_max=127,
        dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, reduce_range=False
    )
    return quant.QConfig(activation=activation_qconfig, weight=weight_qconfig)


def train_and_save(
        in_dim,
        dim,
        layers,
        window_size,
        enable_quant,
        dataset_dir,
        output_dir,
        batch_size,
        num_epochs,
        num_workers,
        lr,
        weight_decay,
        max_norm,
        gpu,
        seed,
        lsb_eliminate=0,
):
    os.makedirs(output_dir, exist_ok=True)

    torch.cuda.set_device(gpu)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger_path = os.path.join(output_dir, 'log')
    os.makedirs(logger_path, exist_ok=True)
    logger = SummaryWriter(logger_path)

    dataset = SignalDataset(
        data_path=dataset_dir,
        is_validate=False,
        read_limit=None,
        train_ratio=0.99,
        seed=42,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        sampler=None,
    )
    print('Dataset size:', len(dataset))
    print('Training steps:', len(train_loader))

    model = QAT_MLP_Compression(
        in_dim=in_dim,
        dim=dim,
        num_dec_layers=layers,
        vocab_size=257,
        max_context_len=1024,
        window_size=window_size,
        enable_quant=enable_quant,
        lsb_eliminate=lsb_eliminate,
    )
    model = model.cuda(gpu)
    model.train()

    if enable_quant:
        model.decoder.fuse_model()
        model.qconfig = get_qat_config()
        quant.prepare_qat(model, inplace=True)

    optimizer = torch.optim.AdamW(
        params=filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler_fn = linear_warmup_cosine_decay(
        warmup_steps=500,
    )
    scheduler = scheduler_fn(optimizer, train_loader, num_epochs, 0)

    scaler = torch.GradScaler('cuda', enabled=True)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        step_count = 0

        for step, batched_data in tqdm(enumerate(train_loader), total=len(train_loader), desc='Epoch {}'.format(epoch)):
            global_step = epoch * len(train_loader) + step

            optimizer.zero_grad()

            batched_data = batched_data.cuda(gpu, non_blocking=True)

            with torch.autocast('cuda', enabled=True, dtype=torch.float16):
                loss, _ = model(batched_data)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                max_norm=max_norm
            ).item()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            step_count += 1

            logger.add_scalar('train/loss', loss.item(), global_step + 1)
            logger.add_scalar('train/grad_norm', grad_norm, global_step + 1)
            logger.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step + 1)
            logger.add_scalar('train/scaler_factor', scaler.get_scale(), global_step + 1)

        print('GPU_{} epoch_{} train_loss = {}'.format(gpu, epoch + 1, total_loss / step_count))

        state_dict_dir = os.path.join(output_dir, 'weights')
        os.makedirs(state_dict_dir, exist_ok=True)

        model.eval()
        save_path = os.path.join(state_dict_dir, 'epoch_{}.checkpoint.pth'.format(epoch + 1))
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}\n")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="train the SigComp MLP entropy model",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--in-dim', type=int, default=128,
                        help='Token embedding dimension')
    parser.add_argument('--dim', type=int, default=256,
                        help='Decoder base dimension; hidden linear layers use twice this width')
    parser.add_argument('--layers', type=int, default=8,
                        help='Number of hidden MLP decoder layers')
    parser.add_argument('--window-size', type=int, default=64,
                        help='Context length in preceding tokens')
    parser.add_argument('--quant', action='store_true', default=False,
                        help='Enable quantization-aware training with fake-quantization observers')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Training dataset directory containing signal_raw.npy, as produced by the dataset command')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for weights/, TensorBoard log/, and config.txt; created if needed')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Signal slices per training batch')
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='Number of complete training passes through the dataset')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='Data-loader subprocess count; 0 loads data in the training process')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='Initial AdamW learning rate before the warmup/cosine schedule')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='AdamW weight-decay coefficient')
    parser.add_argument('--max-norm', type=float, default=1.0,
                        help='Maximum L2 norm for gradient clipping')
    parser.add_argument('--gpu', type=int, default=0,
                        help='Logical training GPU index within the visible CUDA/ROCm devices')
    parser.add_argument('--lsb-eliminate', type=int, default=0,
                        help='Lossy LSB reduction bits (0=lossless). '
                             'Apply BLOW5 lossy bit-elimination to signals '
                             'before delta-zigzag so the model learns the reduced distribution.')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'config.txt'), 'w') as f:
        f.write(
            ' '.join(['python'] + sys.argv) + '\n'
        )

    train_and_save(
        in_dim=args.in_dim,
        dim=args.dim,
        layers=args.layers,
        window_size=args.window_size,
        enable_quant=args.quant,
        dataset_dir=args.dataset,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_norm=args.max_norm,
        gpu=args.gpu,
        seed=42,
        lsb_eliminate=args.lsb_eliminate,
    )

