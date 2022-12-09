import sys
sys.path.append("/home/anthonydavidfuller/xla/test")

import args_parse
from functools import partial
import torch
import torch.nn as nn
import random
import torch.optim as optim
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils
from he_mae import MaskedAutoencoderViT
from einops import rearrange, repeat
from torch.utils.data import TensorDataset
import math
PT_XLA_DEBUG=1


FLAGS = args_parse.parse_common_options(
    datadir='/tmp/imagenet',
    batch_size=None,
    num_epochs=None,
    momentum=None,
    lr=None,
    target_accuracy=None,
    profiler_port=9012,
    opts=MODEL_OPTS.items(),
)


def gauss_noise_tensor(img, sigma=0.05):
    out = img + sigma * torch.randn_like(img)
    return out


# Only instantiate model weights once in memory.
WRAPPED_MODEL = xmp.MpModelWrapper(MaskedAutoencoderViT(
    img_size=112, in_chans=15, patch_size=8, embed_dim=768,
    depth=12, num_heads=12, decoder_embed_dim=512, decoder_depth=2,
    decoder_num_heads=16, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6)))

SERIAL_EXEC = xmp.MpSerialExecutor()

def _train_update(device, step, loss, tracker, epoch, writer):
    test_utils.print_training_update(
        device,
        step,
        loss.item(),
        tracker.rate(),
        tracker.global_rate(),
        epoch,
        summary_writer=writer)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def adjust_learning_rate(_steps, warmup=10_240, max_LR=6e-4, timescale=10_000):
    if _steps < warmup:
        return max_LR * _steps / warmup
    else:
        shift = timescale - warmup
        return max_LR / math.sqrt((_steps + shift) / timescale)

save_steps = [102_400, 204_800]

def train_imagenet():
    print('==> Preparing data..')
    print(f'World size: {xm.xrt_world_size()}')

    def get_train_loader(file_numbers):
        print(f'LOADING FILES: {file_numbers} Device: {xm.xla_device()}')
        all_images = torch.load(f'/mnt/disks/persist/to_tpu/{file_numbers}.pt').float()
        print(f'DATA SHAPE: {all_images.shape}')
        labels = torch.zeros(all_images.shape[0]).view(all_images.shape[0])
        return TensorDataset(all_images, labels)


    torch.manual_seed(42)
    device = xm.xla_device()
    model = WRAPPED_MODEL.to(device)

    writer = None
    if xm.is_master_ordinal():
        writer = test_utils.get_summary_writer(FLAGS.logdir)

    start_lr = 0
    optimizer = optim.AdamW(
        model.parameters(),
        betas=(0.9, 0.95),
        weight_decay=0.05,
        lr=start_lr)

    def train_loop_fn(loader, _epoch, _M=1):
        tracker = xm.RateTracker()
        model.train()
        dev = xm.xla_device()
        train_losses = []
        _steps = 0

        for data, target in loader:
            # data is shape (BSZ, 15, 112, 112)
            optimizer.zero_grad()

            if _M > 1:
                data = repeat(data, 'b c h w -> (M b) c h w', M=M)

            # Run through MAE model
            loss, _, _ = model(imgs=gauss_noise_tensor(data), clean_imgs=data, mask_ratio=0.75)
            train_losses.append(loss.view(1).detach())

            loss.backward()
            xm.optimizer_step(optimizer)
            tracker.add(FLAGS.batch_size)

            if (_steps+1) % 20 == 0:
                xm.add_step_closure(_train_update, args=(device, _steps, loss, tracker, _epoch, writer))

            _steps += 1

        avg_loss = torch.cat(train_losses, dim=0).mean()
        return avg_loss, _steps

    batch_input = 2
    M = int(128/batch_input)
    print(f'M={M}')
    step_counter = 0

    for epoch in range(1_000):
        file_ids = [x for x in range(1, 101)]
        print(len(file_ids))
        random.shuffle(file_ids)
        xm.master_print('Epoch {} train begin {}'.format(epoch, test_utils.now()))

        # TRAIN EPOCH
        for file_id in file_ids:
            train_dataset = get_train_loader(file_numbers=file_id)
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_input,
                sampler=None,
                drop_last=FLAGS.drop_last,
                shuffle=True,
                num_workers=FLAGS.num_workers)
            train_device_loader = pl.MpDeviceLoader(train_loader, device)

            # TRAIN FILE
            file_train_loss, steps_taken = train_loop_fn(loader=train_device_loader, _epoch=epoch, _M=M)
            step_counter += steps_taken

            # UPDATE LEARNING RATE
            set_LR = adjust_learning_rate(step_counter)  # get LR for this epoch
            for g in optimizer.param_groups:
                g['lr'] = set_LR  # update
            current_lr = get_lr(optimizer)  # confirm

            print(epoch, step_counter, current_lr, xm.xla_device(), file_id, file_train_loss)

            if step_counter in save_steps:
                save_obj = {'model': model.state_dict(),
                            'opt': optimizer.state_dict(),
                            'steps_taken': step_counter}
                save_name = f'/home/anthonydavidfuller/M{M}_{step_counter}.pt'

                xm.master_print(f'Saving checkpoint: {save_name}')
                xm.save(save_obj, save_name)

                if step_counter == save_steps[-1]:
                    xm.master_print(f'STOPPING: {save_name}')
                    exit()

        xm.master_print('Epoch {} train end {}'.format(epoch, test_utils.now()))
        xm.master_print(met.metrics_report())


def _mp_fn(index, flags):
    global FLAGS
    FLAGS = flags
    torch.set_default_tensor_type('torch.FloatTensor')
    train_imagenet()


if __name__ == '__main__':
    xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=FLAGS.num_cores)