import os
import sys
import random
import numpy as np
import torch
import torch.ao.quantization as quant
from tqdm import tqdm
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from .model import QAT_MLP_Compression
from .dataset import SignalDataset


if __name__ == "__main__":
    parser = ArgumentParser(
        description="run post-training quantization for a trained model",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--calibration-dataset', type=str, default=None, required=True, help="Calibration dataset path")
    parser.add_argument('--FP32-model', type=str, default=None, required=True, help="FP32 model checkpoint path")
    parser.add_argument('--output-model', type=str, default=None, required=True, help="Output INT8 model checkpoint path")
    parser.add_argument('--batch-size', type=int, default=64, )
    parser.add_argument('--window-size', type=int, default=64, )
    parser.add_argument('--in-dim', type=int, default=128, )
    parser.add_argument('--dim', type=int, default=256, )
    parser.add_argument('--num-layers', type=int, default=8, )
    parser.add_argument('--lsb-eliminate', type=int, default=0, )

    args = parser.parse_args()
    quant_model_path = args.FP32_model
    out_quant_model_path = args.output_model
    batch_size = args.batch_size

    dataset = SignalDataset(
        data_path=args.calibration_dataset,
        is_validate=False,
        read_limit=None,
        train_ratio=0.99,
        seed=42,
    )
    calib_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=1,
        pin_memory=True,
        sampler=None,
    )

    quant_model = QAT_MLP_Compression(
        in_dim=args.in_dim,
        dim=args.dim,
        num_dec_layers=args.num_layers,
        vocab_size=257,
        max_context_len=1024,
        window_size=args.window_size,
        enable_quant=True,
        lsb_eliminate=args.lsb_eliminate,
    )

    quant_model.load_state_dict(torch.load(quant_model_path, weights_only=True))

    quant_model.eval()

    quant_model.decoder.fuse_model()

    qconfig = quant.QConfig(
        activation=quant.MovingAverageMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, reduce_range=False),
        weight=quant.MovingAverageMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric),
    )
    quant_model.qconfig = qconfig

    quant_model.decoder.embedding_linear.qconfig = None

    quant.prepare(quant_model, inplace=True)

    total_loss = 0
    step_count = 0
    max_step = 30
    with torch.no_grad():
        for step, inputs in tqdm(enumerate(calib_dataloader)):
            loss, _ = quant_model(inputs)
            print(f'Step {step} Loss = {loss.item()}')
            total_loss += loss.item()
            step_count += 1
            if step_count >= max_step:
                break
    print('Total loss:', total_loss / step_count)

    quant.convert(quant_model, inplace=True)
    print(quant_model)

    torch.save(quant_model.state_dict(), out_quant_model_path)
    print('Done')

