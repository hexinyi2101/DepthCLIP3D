import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel

from training.distributed import is_master
from training.precision import get_autocast

from d_cls.zeroshot_cls import evaluate_d_cls

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype, CLIP, CustomTextCLIP
from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }

def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def encode_rgb(rgbs, modelrgb, device, normalize: bool = False):   
    inputs = {'image': {'pixel_values': rgbs.to(device).float()}}

    with torch.no_grad():
        embeddings,patch_features = modelrgb(inputs)
    #torch.save(patch_features, 'patch_features.pt')
    #torch.save(rgbs, 'rgbs.pt')

    return embeddings['image'] / embeddings['image'].norm(p=2, dim=-1, keepdim=True) if normalize else embeddings['images'],patch_features

def select_tokens(rgb_features_dict, patch_indices_keep, device):

    rgb_features = rgb_features_dict['image']
    batch_size, num_tokens, feature_dim = rgb_features.size()

    # 确保 patch_indices_keep 在相同设备上
    patch_indices_keep = patch_indices_keep.to(device)  # 移动到相同设备

    cls_token = rgb_features[:, 0, :].unsqueeze(1)  # Shape: [batch_size, 1, feature_dim]

    # 确保patch_indices_keep的索引适用于从第一个非CLS token（即索引1开始的tokens）开始的范围
    adjusted_indices = patch_indices_keep  # 索引调整，因为第一个token（CLS token）在位置0
    selected_tokens = torch.gather(rgb_features[:, 1:, :], 1, adjusted_indices.unsqueeze(2).expand(-1, -1, feature_dim))
    result_features = torch.cat([cls_token, selected_tokens], dim=1)  # Shape: [batch_size, num_keep_tokens + 1, feature_dim]
    result_features = result_features.to(device)

    result_features = {'image': result_features}
    # 返回一个新的字典，包含处理后的张量
    print("-------------",result_features)
    return  result_features
def apply_patch_indices(rgbpatch_features, patch_indices_keep, device, exclude_first_token=True):
    # 如果排除首个token，则先分割
    rgbpatch_features = rgbpatch_features['image']
    patch_indices_keep = patch_indices_keep.to(device)  # 移动到相同设备

    if exclude_first_token:
        cls_tokens_rgb, rgbpatch_features = rgbpatch_features[:, :1], rgbpatch_features[:, 1:]
    
    batch = rgbpatch_features.size()[0]
    batch_indices = torch.arange(batch)
    batch_indices = batch_indices[..., None]
    rgbpatch_features = rgbpatch_features[batch_indices, patch_indices_keep].to(device)

    # 如果排除了首个token，现在将其重新连接
    if exclude_first_token:
        rgbpatch_features = torch.cat((cls_tokens_rgb, rgbpatch_features), dim=1)

    return {'image': rgbpatch_features}

def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, modelrgb, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)


    model.train()
    if args.distill:
        dist_model.eval()

    data[f'{args.clip_type}_pt'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data[f'{args.clip_type}_pt'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_input_ids, accum_attention_mask, accum_features = [], [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        # print(f"batch:{batch}")
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        images, input_ids, attention_mask, rgbs = batch
        #torch.save(attention_mask, 'attention_mask.pt')
        # print(f" images:{len(images)}, input_ids:{len(input_ids)}, attention_mask :{len(attention_mask)},rgbs:{rgbs}")
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        rgbs = rgbs.to(device=device, dtype=input_dtype, non_blocking=True)
        input_ids = input_ids.to(device=device, non_blocking=True)
        attention_mask = attention_mask.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                rgb_features,rgbpatch_features = encode_rgb(rgbs, modelrgb, device, normalize=True) if rgbs is not None else None
                model_out = model(images, rgb_features, rgbpatch_features,input_ids, attention_mask)
                #print(len(model_out["image_features"][1]))
                logit_scale = model_out["logit_scale"]
                if args.distill:
                    with torch.no_grad():
                        dist_model_out = dist_model(images, input_ids, attention_mask)
                    model_out.update({f'dist_{k}' : v for k, v in dist_model_out.items()})
                rgbs = model_out["rgb_features"]
                depth_token_feature = model_out["img_token_feature"]
                patch_indices_keep = model_out["patch_indices_keep"]
                rgbpatch_features = model_out["rgb_token_feature"]
                # print(patch_indices_keep,depth_token_feature,rgbpatch_features)
                model_out.pop('rgb_features', None)  # 移除 'rgb_features' 键（如果存在）
                model_out.pop('img_token_feature', None)  # 移除 'rgb_features' 键（如果存在）
                model_out.pop('patch_indices_keep', None)  # 移除 'patch_indices_keep' 键（如果存在）
                model_out.pop('rgb_token_feature', None)  # 移除 'rgb_token_feature' 键（如果存在）
                #torch.save(rgbs, 'rgbsintrain.pt')
                #torch.save(depth_token_feature, 'depth_token_feature_intrain.pt')
                rgbpatch_features = apply_patch_indices(rgbpatch_features, patch_indices_keep, device=device, exclude_first_token=True)
                losses = loss(**model_out, rgbs=rgbs, rgbtoken=rgbpatch_features, depthtoken=depth_token_feature,output_dict=True)

                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, input_ids, attention_mask)
                    model_out.pop("logit_scale")
                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_input_ids.append(input_ids)
                accum_attention_mask.append(attention_mask)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                input_ids = accum_input_ids[j]
                attention_mask = accum_attention_mask[j]
                with autocast():
                    model_out = model(images, input_ids, attention_mask)
                    logit_scale = model_out.pop("logit_scale")
                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] +  [model_out[key]] + accumulated[j + 1:])
                    losses = loss(**inputs, logit_scale=logit_scale, output_dict=True)
                    del inputs
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss
                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_input_ids, accum_attention_mask, accum_features = [], [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            # if args.add_time_attn:
            #     if hasattr(model, 'module'):
            #         t_gate = [[F.sigmoid(m.t_attn_gate).detach().item(), F.sigmoid(m.t_ffn_gate).detach().item()] for m in model.module.vision_model.encoder.layers]
            #     else:
            #         t_gate = [[F.sigmoid(m.t_attn_gate).detach().item(), F.sigmoid(m.t_ffn_gate).detach().item()] for m in model.vision_model.encoder.layers]
            #     t_attn_gate, t_ffn_gate = list(zip(*t_gate))
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            # if args.add_time_attn:
            #     logging.info(
            #         f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
            #         f"Data (t): {data_time_m.avg:.3f} "
            #         f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
            #         f"LR: {optimizer.param_groups[0]['lr']:5f} "
            #         f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log +
            #         f"\nt_attn_gate: {[round(i, 2) for i in t_attn_gate]}\nt_ffn_gate: {[round(i, 2) for i in t_ffn_gate]}\n"
            #     )
            # else:
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )


            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            log_data.update({name:val.val for name,val in losses_m.items()})
            # if args.add_time_attn:
            #     log_data.update({f'layer_{i}_t_attn_gate': attn for i, attn in enumerate(t_attn_gate)})
            #     log_data.update({f'layer_{i}_t_ffn_gate': ffn for i, ffn in enumerate(t_ffn_gate)})

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
        '''
        #以下为epoch内部的测试
        import copy
        from data.build_datasets import get_data

        args_copy = copy.deepcopy(args)  # 创建args的一个深拷贝副本
        args_copy.val_d_cls_data = ["SUNRGBD"]
        args_copy.do_train = False
        data_sun = get_data(args_copy, epoch=epoch)

        args_copy = copy.deepcopy(args)  # 创建args的一个深拷贝副本
        args_copy.val_d_cls_data = ["NYUV2"]
        args_copy.do_train = False
        data_nyu = get_data(args_copy, epoch=epoch)  

        args_copy = copy.deepcopy(args)  # 创建args的一个深拷贝副本
        args_copy.val_d_cls_data = ["shapenet12"]
        args_copy.do_train = False
        data_shape = get_data(args_copy, epoch=epoch)  

        processed_samples_count += args.batch_size * args.world_size  # 如果是分布式训练
        if processed_samples_count // 10240 > (processed_samples_count - args.batch_size * args.world_size) // 10240:
            # 如果达到了10240的倍数，则使用当前模型进行测试
            model.eval()  # 设置模型为评估模式
            with torch.no_grad():  # 关闭梯度计算
                for sub_data in data_sun['d_cls']:
                    evaluate_d_cls(model, sub_data, epoch, args, tb_writer)
                
                for sub_data in data_nyu['d_cls']:
                    evaluate_d_cls(model, sub_data, epoch, args, tb_writer)
                
                for sub_data in data_shape['d_cls']:
                    evaluate_d_cls(model, sub_data, epoch, args, tb_writer)
                model.train()  # 测试完成后将模型设置回训练模式'''
    # end for
