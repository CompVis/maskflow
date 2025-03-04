import torch
from einops import repeat
import sys
import os
from omegaconf import OmegaConf
from einops import rearrange
from tqdm import tqdm
from diffusers import AutoencoderKL
import wandb
import numpy as np
import random
import torch
import imageio
import uuid
import torch.nn.functional as F
from datetime import datetime
from dataloader_utils import get_dataloader
from einops import repeat
from torchvision.utils import draw_bounding_boxes
import torch.distributed as dist


def out2img(samples):
    return torch.clamp(127.5 * samples + 128.00, 0, 255).to(
        dtype=torch.uint8, device="cuda"
    )


def get_png_name(sample_root, vis_task, _time_str, postfix=None):
    if postfix is None:
        postfix = ""
    filename = vis_task + "_" + _time_str + "_" + postfix + ".png"
    return os.path.join(sample_root, filename)

def print_rank_0(*args, **kwargs):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            # print(*args, **kwargs)
            try:
                logging.info(*args, **kwargs)
            except:
                print(*args, **kwargs)
    else:
        print(*args, **kwargs)

def wandb_visual_dict(wandb_key, visual_tensor, is_video, num=16, captions=None):
    if captions is None:
        captions = ["null caption" for _ in range(num)]
    if is_video:
        b, t, c, w, h = visual_tensor.shape
        visual_tensor = visual_tensor.cpu().numpy()
        return {
            wandb_key: wandb.Video(visual_tensor[:num]),
        }
    else:
        b, c, w, h = visual_tensor.shape
        return {
            wandb_key: wandb.Image(array2grid_pixel(visual_tensor[:num])),
        }

def get_max_ckpt_from_dir(dir_path):
    dir_path = os.path.join(dir_path, "checkpoints")
    # Define the pattern to match
    pattern = r"(\d+)\.pt"

    
    max_step = -1
    max_step_file = None
    for filename in os.listdir(dir_path):
        match = re.match(pattern, filename)
        if match:
            step = int(match.group(1))
            if step > max_step:
                max_step = step
                max_step_file = filename

    if max_step_file is None:
        raise ValueError(f"No checkpoint files found in {dir_path}")
    else:
        print(
            f"Found checkpoint file {max_step_file} with step {max_step} from {dir_path}"
        )
        return os.path.join(dir_path, max_step_file)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


# def vis_vq(
#     args,
#     model,
#     tokenizer_encode_fn,
#     tokenizer_decode_fn,
#     vae,
#     accelerator,
#     device,
#     sample_fn,
#     vis_task,
#     skip_data_loader=True,
#     # local_bs=4,
#     local_bs=4,
#     sample_root="sample_vis/",
# ):

#     assert vis_task in [
#         "cfg_range",
#         "nfe_range",
#         "vis_chain",
#         "temp_range",
#         "vis_class",
#         "vis_maxchain",
#         "vis_inpaint",
#     ]
#     if accelerator.is_main_process:
#         wandb_run = wandb.init(project="discretediffusion_papervis", name=vis_task)

#     if False:  # demo for running visualization
#         if args.data.num_classes > 0:
#             y = torch.randint(0, args.data.num_classes - 1, (local_bs,), device=device)
#         else:
#             y = None
#         model_kwargs = dict(vis_task=vis_task, y=y)
#         model_fn = model.forward_without_cfg

#         if not skip_data_loader:
#             gts = next(generator)
#             _sample_size = vq_get_sample_size(len(gts), args)

#             with torch.no_grad():
#                 indices_chains, mask_chains = sample_fn(
#                     _sample_size, model_fn, **model_kwargs
#                 )
#                 _indices = indices_chains[-1]
#                 samples = tokenizer_decode_fn(_indices)

#             gts = gts[: len(samples)]
#         else:
#             _sample_size = vq_get_sample_size(len(y), args)

#             with torch.no_grad():
#                 indices_chains, mask_chains = sample_fn(
#                     _sample_size, model_fn, **model_kwargs
#                 )
#                 _indices = indices_chains[-1]
#                 samples = tokenizer_decode_fn(_indices)
#             gts = samples.clone()

#         sam_4fid, gts_4fid = samples, gts
#         gts_4fid = accelerator.gather(gts_4fid.to(device))
#         sam_4fid = accelerator.gather(sam_4fid.to(device))
#         accelerator.wait_for_everyone()
#     elif vis_task == "vis_class":
#         if args.data.num_classes > 0:
#             # https://github.com/valeoai/Maskgit-pytorch/blob/b0b2b3cc11cffd0b159f22dc1c6e73a7e8b53db3/Trainer/vit.py#L338C19-L338C109
#             # goldfish, chicken, tiger cat, hourglass, ship, dog, race car, airliner, teddy bear, random

#             y = [
#                 75,
#                 124,
#                 189,
#                 228,
#                 232,
#                 233,
#                 281,
#                 282,
#                 288,
#                 random.randint(0, 999),
#             ]
#             vis_class_num = len(y)
#             y = y * 2  # (nb_sample // 10)
#             y = torch.LongTensor(y).to(device)

#         else:
#             y = None
#         ###########################################################
#         model_kwargs = dict(vis_task=vis_task, y=y, return_chains=2)  # return chains
#         model_fn = model.forward_without_cfg
#         ##########################################################
#         _sample_size = vq_get_sample_size(len(y), args)
#         with torch.no_grad():
#             indices_chains, mask_chains = sample_fn(
#                 _sample_size, model_fn, **model_kwargs
#             )

#             _indices = indices_chains[-1]
#             samples = tokenizer_decode_fn(_indices)

#         sam_4fid = accelerator.gather(samples.to(device))
#         accelerator.wait_for_everyone()
#         _time_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
#         if accelerator.is_main_process:
#             _sample_root = get_png_name(sample_root, vis_task, _time_str)
#             imageio.imwrite(
#                 _sample_root, array2grid_pixel_ncol(sam_4fid, ncol=vis_class_num)
#             )
#             print(f"save to {_sample_root}")
#             wandb_list = []
#             for i, _wandb_image in enumerate(sam_4fid):
#                 wandb_list.append(wandb.Image(array2row_pixel(_wandb_image)))
#             wandb.log({vis_task: wandb_list})
#     elif vis_task == "vis_inpaint":
#         exampler_per_mask = 6
#         assert args.dynamic.discretefm.sampler == "maskgit"
#         loader = get_dataloader(args)
#         loader = accelerator.prepare(loader)
#         data_generator, _generator, caption_generator = vq_get_generator(
#             args, device, loader, 0, vae
#         )
#         ###########################################################
#         x, y = next(data_generator)
#         x_pixel = tokenizer_decode_fn(x)

#         paint_mask, pixel_with_redbbox = get_inpaint_mask(
#             x, mask_ratio=0.4, downsample_ratio=8, x_pixel=x_pixel
#         )

#         x_indices_broken = x.clone()
#         x_indices_broken[paint_mask == 1] = args.tokenizer.mask_token_id
#         model_fn = model.forward_without_cfg
#         init_code = x_indices_broken
#         y = repeat(y, "b -> (b ss)", ss=exampler_per_mask)
#         init_code = repeat(init_code, "b w h -> (b ss) w h", ss=exampler_per_mask)
#         model_kwargs = dict(
#             vis_task=vis_task,
#             y=y,
#             return_chains=1,  # 1: chain, 2: maxchain
#             init_code=init_code,
#         )
#         ##########################################################
#         _sample_size = vq_get_sample_size(len(y), args)
#         with torch.no_grad():
#             indices_chains, mask_chains = sample_fn(
#                 _sample_size, model_fn, **model_kwargs
#             )
#             # chains_num, b, *latent_size = indices_chains.shape
#             # _indices = indices_chains.reshape(-1, *latent_size)
#             _indices = indices_chains[-1]

#             samples = tokenizer_decode_fn(_indices)
#             # [B,C,W,H]
#             ###########################################################

#         sam_4fid = samples
#         sam_4fid = accelerator.gather(sam_4fid.to(device))
#         sam_4fid = rearrange(
#             sam_4fid, "(b ss) c h w -> b ss c h w", ss=exampler_per_mask
#         )
#         pixel_with_redbbox = repeat(pixel_with_redbbox, "b c w h -> b 1 c w h")
#         all_imgs = torch.cat([pixel_with_redbbox, sam_4fid], dim=1)  # [B,3,C,W,H]

#         accelerator.wait_for_everyone()
#         _time_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
#         if accelerator.is_main_process:
#             wandb_list = []
#             for i, _wandb_image in enumerate(all_imgs):
#                 _sample_root = get_png_name(sample_root, vis_task, _time_str, f"img{i}")
#                 wandb_list.append(wandb.Image(array2row_pixel(_wandb_image)))
#                 imageio.imwrite(_sample_root, array2row_pixel(_wandb_image))
#                 print(f"save img to {_sample_root}")
#             wandb.log({vis_task: wandb_list})
#     elif vis_task == "vis_chain":
#         set_seed(666)
#         if args.data.num_classes > 0:
#             y = torch.randint(0, args.data.num_classes - 1, (local_bs,), device=device)
#         else:
#             y = None
#         ###########################################################
#         model_kwargs = dict(vis_task=vis_task, y=y, return_chains=1)  # return chains
#         model_fn = model.forward_without_cfg
#         ###########################################################

#         if not skip_data_loader:
#             gts = next(generator)
#             _sample_size = vq_get_sample_size(len(gts), args)

#             with torch.no_grad():
#                 indices_chains, mask_chains = sample_fn(
#                     _sample_size, model_fn, **model_kwargs
#                 )
#                 _indices = indices_chains[-1]
#                 samples = tokenizer_decode_fn(_indices)

#             gts = gts[: len(samples)]
#         else:
#             _sample_size = vq_get_sample_size(local_bs, args)
#             with torch.no_grad():
#                 indices_chains, mask_chains = sample_fn(
#                     _sample_size, model_fn, **model_kwargs
#                 )
#                 chains_num, b, *latent_size = indices_chains.shape
#                 _indices = indices_chains.reshape(-1, *latent_size)

#                 samples = tokenizer_decode_fn(_indices)
#                 _, *image_size = samples.shape
#                 samples = samples.reshape(chains_num, b, *image_size)
#                 samples = rearrange(samples, "t b c h w -> b t c h w")
#                 ###########################################################
#                 if len(_sample_size) == 4:
#                     mask_chains = (
#                         rearrange(mask_chains * 255.0, "t b c w h-> (b t) c w h")
#                     ).to(torch.uint8)
#                     mask_chains = F.interpolate(mask_chains, (256, 256), mode="nearest")
#                     mask_chains = rearrange(
#                         mask_chains, "(b t) c h w -> b t c h w", b=b
#                     )
#                 elif len(_sample_size) == 3:
#                     mask_chains = (
#                         rearrange(mask_chains * 255.0, "t b w h -> (b t) 1 w h")
#                     ).to(torch.uint8)
#                     mask_chains = F.interpolate(mask_chains, (256, 256), mode="nearest")
#                     mask_chains = rearrange(
#                         mask_chains, "(b t) 1 h w -> b t 1 h w", b=b
#                     )

#                 elif len(_sample_size) == 2:
#                     mask_chains[mask_chains == 0] = 0.88
#                     mask_chains = (
#                         rearrange(mask_chains * 255.0, "t b l -> (b t) 1 1 l")
#                     ).to(torch.uint8)
#                     mask_chains = F.interpolate(mask_chains, (256, 256), mode="nearest")
#                     mask_chains = rearrange(
#                         mask_chains, "(b t) 1 h w -> b t 1 h w", b=b
#                     )
#                 else:
#                     raise ValueError(
#                         f"_sample_size must be 2, 3, or 4, but got {len(_sample_size)}"
#                     )

#         sam_4fid = samples
#         sam_4fid = accelerator.gather(sam_4fid.to(device))
#         mask_chains = accelerator.gather(mask_chains.to(device))
#         accelerator.wait_for_everyone()
#         _time_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
#         if accelerator.is_main_process:
#             wandb_list = []
#             for i, (_wandb_image, _mask_chain) in enumerate(zip(sam_4fid, mask_chains)):
#                 _sample_root = get_png_name(sample_root, vis_task, _time_str, f"img{i}")
#                 _mask_root = get_png_name(sample_root, vis_task, _time_str, f"mask{i}")
#                 wandb_list.append(wandb.Image(array2row_pixel(_wandb_image)))
#                 wandb_list.append(wandb.Image(array2row_pixel(_mask_chain)))
#                 imageio.imwrite(_sample_root, array2row_pixel(_wandb_image))
#                 imageio.imwrite(_mask_root, array2row_pixel(_mask_chain))
#                 print(f"save img to {_sample_root}")
#                 print(f"save mask to {_mask_root}")
#             wandb.log({vis_task: wandb_list})
#     elif vis_task == "vis_maxchain":
#         set_seed(666)
#         if args.data.num_classes > 0:
#             y = torch.randint(0, args.data.num_classes - 1, (local_bs,), device=device)
#         else:
#             y = None
#         ###########################################################
#         model_kwargs = dict(vis_task=vis_task, y=y, return_chains=2)  # return chains
#         # 1: normal chain; 2: max chain
#         model_fn = model.forward_without_cfg
#         ###########################################################

#         if not skip_data_loader:
#             gts = next(generator)
#             _sample_size = vq_get_sample_size(len(gts), args)

#             with torch.no_grad():
#                 indices_chains, mask_chains = sample_fn(
#                     _sample_size, model_fn, **model_kwargs
#                 )
#                 _indices = indices_chains[-1]
#                 samples = tokenizer_decode_fn(_indices)

#             gts = gts[: len(samples)]
#         else:
#             _sample_size = vq_get_sample_size(len(y), args)
#             with torch.no_grad():
#                 indices_chains, mask_chains = sample_fn(
#                     _sample_size, model_fn, **model_kwargs
#                 )
#                 chains_num, b, *latent_size = indices_chains.shape
#                 _indices = indices_chains.reshape(-1, *latent_size)
#                 samples = tokenizer_decode_fn(_indices)
#                 _, *image_size = samples.shape
#                 samples = samples.reshape(chains_num, b, *image_size)
#                 samples = rearrange(samples, "t b c h w -> b t c h w")

#         sam_4fid = samples
#         sam_4fid = accelerator.gather(sam_4fid.to(device))
#         accelerator.wait_for_everyone()
#         _time_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
#         if accelerator.is_main_process:
#             wandb_list = []
#             for i, _wandb_image in enumerate(sam_4fid):
#                 _sample_root = get_png_name(sample_root, vis_task, _time_str, i)
#                 wandb_list.append(wandb.Image(array2row_pixel(_wandb_image)))
#                 imageio.imwrite(_sample_root, array2row_pixel(_wandb_image))
#                 print(f"save to {_sample_root}")
#             wandb.log({vis_task: wandb_list})

#     elif vis_task == "cfg_range":
#         if args.data.num_classes > 0:
#             # y = torch.randint(0, args.data.num_classes - 1, (local_bs,), device=device)
#             y = torch.randint(100, 101, (local_bs,), device=device)
#         else:
#             y = None
#         ###########################################################
#         model_kwargs = dict(
#             vis_task=vis_task, y=y, return_chains=False
#         )  # return chains
#         model_fn = model.forward_with_cfg
#         ###########################################################

#         cfg_vis_list = []
#         for _cfg in [0.1, 0.3, 0.5, 0.7, 0.9]:
#             # for _cfg in [0.3, 0.5]:
#             model_kwargs["cfg_scale"] = _cfg
#             set_seed(666)
#             if not skip_data_loader:
#                 gts = next(generator)
#                 _sample_size = vq_get_sample_size(len(gts), args)
#                 with torch.no_grad():
#                     indices_chains, mask_chains = sample_fn(
#                         _sample_size, model_fn, **model_kwargs
#                     )
#                     _indices = indices_chains[-1]
#                     samples = tokenizer_decode_fn(_indices)

#                 gts = gts[: len(samples)]
#             else:
#                 _sample_size = vq_get_sample_size(len(y), args)
#                 with torch.no_grad():
#                     _indices, mask_chains = sample_fn(
#                         _sample_size, model_fn, **model_kwargs
#                     )
#                     samples = tokenizer_decode_fn(_indices)

#             sam_4fid = samples
#             sam_4fid = accelerator.gather(sam_4fid.to(device))
#             accelerator.wait_for_everyone()
#             cfg_vis_list.append(sam_4fid)
#         cfg_vis_list = torch.stack(cfg_vis_list, dim=0)
#         cfg_vis_list = rearrange(cfg_vis_list, "cfg bs c h w -> bs cfg c h w")
#         wandb_list = []
#         _time_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
#         if accelerator.is_main_process:
#             for i, _wandb_image in enumerate(cfg_vis_list):
#                 _sample_root = get_png_name(sample_root, vis_task, _time_str, i)
#                 wandb_list.append(wandb.Image(array2row_pixel(_wandb_image)))
#                 imageio.imwrite(_sample_root, array2row_pixel(_wandb_image))
#                 print(f"save to {_sample_root}")
#             wandb.log({vis_task: wandb_list})
#     elif vis_task == "nfe_range":
#         if args.data.num_classes > 0:
#             y = torch.randint(0, args.data.num_classes - 1, (local_bs,), device=device)
#         else:
#             y = None
#         ###########################################################
#         model_kwargs = dict(
#             vis_task=vis_task, y=y, return_chains=False
#         )  # return chains
#         model_fn = model.forward_without_cfg
#         ###########################################################

#         cfg_vis_list = []
#         # for _cfg in [0.1, 0.3, 0.5, 0.7, 0.9]:
#         for _step_num in [10, 50]:
#             model_kwargs["step_num"] = _step_num
#             if not skip_data_loader:
#                 gts = next(generator)
#                 _sample_size = vq_get_sample_size(len(gts), args)

#                 with torch.no_grad():
#                     indices_chains, mask_chains = sample_fn(
#                         _sample_size, model_fn, **model_kwargs
#                     )
#                     _indices = indices_chains[-1]
#                     samples = tokenizer_decode_fn(_indices)

#                 gts = gts[: len(samples)]
#             else:
#                 _sample_size = vq_get_sample_size(len(y), args)
#                 with torch.no_grad():
#                     _indices, mask_chains = sample_fn(
#                         _sample_size, model_fn, **model_kwargs
#                     )
#                     samples = tokenizer_decode_fn(_indices)

#             sam_4fid = samples
#             sam_4fid = accelerator.gather(sam_4fid.to(device))
#             accelerator.wait_for_everyone()
#             cfg_vis_list.append(sam_4fid)
#         cfg_vis_list = torch.stack(cfg_vis_list, dim=0)
#         cfg_vis_list = rearrange(cfg_vis_list, "cfg bs c h w -> bs cfg c h w")
#         wandb_list = []
#         _time_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
#         if accelerator.is_main_process:
#             for i, _wandb_image in enumerate(cfg_vis_list):
#                 _sample_root = get_png_name(sample_root, vis_task, _time_str, i)
#                 wandb_list.append(wandb.Image(array2row_pixel(_wandb_image)))
#                 imageio.imwrite(_sample_root, array2row_pixel(_wandb_image))
#                 print(f"save to {_sample_root}")
#             wandb.log({vis_task: wandb_list})

#     elif vis_task == "temp_range":
#         if args.data.num_classes > 0:
#             y = torch.randint(0, args.data.num_classes - 1, (local_bs,), device=device)
#         else:
#             y = None
#         ###########################################################
#         model_kwargs = dict(
#             vis_task=vis_task, y=y, return_chains=False
#         )  # return chains
#         model_fn = model.forward_without_cfg
#         ###########################################################

#         cfg_vis_list = []
#         # for _cfg in [0.1, 0.3, 0.5, 0.7, 0.9]:
#         for _temp in [0.990, 0.995, 1.000]:
#             model_kwargs["temperature"] = _temp
#             if not skip_data_loader:
#                 gts = next(generator)
#                 _sample_size = vq_get_sample_size(len(gts), args)

#                 with torch.no_grad():
#                     indices_chains, mask_chains = sample_fn(
#                         _sample_size, model_fn, **model_kwargs
#                     )
#                     _indices = indices_chains[-1]
#                     samples = tokenizer_decode_fn(_indices)

#                 gts = gts[: len(samples)]
#             else:
#                 _sample_size = vq_get_sample_size(len(y), args)
#                 with torch.no_grad():
#                     _indices, mask_chains = sample_fn(
#                         _sample_size, model_fn, **model_kwargs
#                     )
#                     samples = tokenizer_decode_fn(_indices)

#             sam_4fid = samples
#             sam_4fid = accelerator.gather(sam_4fid.to(device))
#             accelerator.wait_for_everyone()
#             cfg_vis_list.append(sam_4fid)
#         cfg_vis_list = torch.stack(cfg_vis_list, dim=0)
#         cfg_vis_list = rearrange(cfg_vis_list, "cfg bs c h w -> bs cfg c h w")
#         if accelerator.is_main_process:
#             wandb_list = []
#             _time_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
#             for i, _wandb_image in enumerate(cfg_vis_list):
#                 _sample_root = get_png_name(sample_root, vis_task, _time_str, i)
#                 wandb_list.append(wandb.Image(array2row_pixel(_wandb_image)))
#                 imageio.imwrite(_sample_root, array2row_pixel(_wandb_image))
#                 print(f"save to {_sample_root}")
#             wandb.log({vis_task: wandb_list})

#     print("done visualization")


def vq_get_sample_size(bs, args):
    if args.input_tensor_type == "bt":
        return (bs, args.tokenizer.token_len)
    elif args.input_tensor_type == "bwh":
        return (bs, args.tokenizer.latent_size, args.tokenizer.latent_size)
    elif args.input_tensor_type == "bcwh":
        return (
            bs,
            args.tokenizer.in_channels,
            args.tokenizer.latent_size,
            args.tokenizer.latent_size,
        )
    elif args.input_tensor_type == "btwh":
        assert args.data.video_frames > 0, "video_frames must be > 0"
        return (
            bs,
            args.data.video_frames,
            args.tokenizer.latent_size,
            args.tokenizer.latent_size,
        )
    elif args.input_tensor_type == "btcwh":
        assert args.data.video_frames > 0, "video frames must be > 0"
        return (
            bs,
            args.data.video_frames,
            args.data.in_channels,
            args.data.image_size,
            args.data.image_size,
        )
    else:
        raise ValueError(f"Unknown tensor type: {args.input_tensor_type}")


def calculate_top_k_accuracy(logits, targets, target_mask, k=10):
    # Usage:
    # logits: shape (batch_size, sequence_length, vocab_size)
    # targets: shape (batch_size, sequence_length)
    # target_mask: shape (batch_size, sequence_length)
    # Get the top k predictions
    _, top_k_predictions = torch.topk(logits, k, dim=-1)

    # Create a boolean tensor indicating if the true label is in the top k predictions
    matches = torch.eq(
        top_k_predictions, targets.unsqueeze(-1).expand_as(top_k_predictions)
    ).any(dim=-1)

    # Calculate accuracy only for masked positions
    acc = (matches * target_mask).sum().float() / (target_mask.sum() + 1e-7)

    return acc


def vq_get_vae(args, device):
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()
    return vae


def vq_get_dynamic(args, device, is_train=True):

    if args.dynamic.name == "maskflow":
        
        from dynamics.maskflow import (
            FMSampler,
            MGMSampler,
            DiscreteFM,
            Ucoupling,
            LinearScheduler,
            CosineScheduler,
            SigmoidScheduler,
            indices_to_diracp,
            CosineScheduler,
            indices_to_diracp
        )

        encode_fn, decode_fn = vq_get_encoder_decoder(args, device)

        mask_token_id = args.tokenizer.mask_token_id 
        vocab_size = args.tokenizer.vocab_size
        input_tensor_type = args.input_tensor_type

        if args.dynamic.scheduler == "linear":
            scheduler=LinearScheduler() 
        elif args.dynamic.scheduler == "cosine":
            scheduler=CosineScheduler()
        elif args.dynamic.scheduler == "sigmoid":
            scheduler=SigmoidScheduler()
        else:
            raise ValueError(f"Scheduler {args.dynamic.scheduler} not supported!")

        flow_matching = DiscreteFM(
            vocab_size=vocab_size,
            coupling=Ucoupling(mask_token_id=mask_token_id),
            kappa=scheduler,
            objective=args.dynamic.objective,
            noise_level=args.dynamic.noise_level,
            device=device,
            input_tensor_type=input_tensor_type,
            reweigh_loss=args.dynamic.reweigh_loss,
            smoothing_factor=args.dynamic.smooth,
            sampling_timesteps=args.dynamic.sampling_timesteps,
            temperature=args.dynamic.temperature,
            use_fp16=args.dynamic.use_fp16,
            mask_ce=args.dynamic.mask_ce,
            cum_snr_decay=args.dynamic.cum_snr_decay,
            snr_clip=args.dynamic.snr_clip,
            use_fused_snr=args.dynamic.use_fused_snr
        )

        if args.dynamic.sampler == "mgm":
            sampler = MGMSampler(
                mask_token_id=mask_token_id,
                input_tensor_type=input_tensor_type
            )
        else:
            sampler = FMSampler(
                mask_token_id=mask_token_id,
                input_tensor_type=input_tensor_type,
                adaptive_stepsize=args.dynamic.adaptive_stepsize,
            )

        def training_losses_fn(model, x, **model_kwargs):
            batch_size, n_frames = x.shape[:2]
            sampling_eps = 1e-3

            if args.dynamic.noise_level == "random_all":
                t = (1 - sampling_eps) * torch.rand((batch_size, n_frames), device=device) + sampling_eps
            elif args.dynamic.noise_level == "constant":
                t = (1 - sampling_eps) * torch.rand((batch_size, 1), device=device) + sampling_eps
                t = t.expand(batch_size, n_frames)
                t = t.clamp(min=sampling_eps, max=1 - sampling_eps)
            elif args.dynamic.noise_level == "dynamic":
                mode = torch.rand(1).item()
                if mode < 0.8:  
                    t = (1 - sampling_eps) * torch.rand((batch_size, n_frames), device=device) + sampling_eps
                else:  
                    t = 0.5 + 0.5 * torch.rand((batch_size, 1), device=device)
                    t = t.expand(batch_size, n_frames)

            loss_dict = flow_matching.training_losses(
                model=model,
                x=x,
                noise_levels=t
            )
            return loss_dict

        def sample_frames_from_batch(args, n_frames):
            train_loader = get_dataloader(args)
            data_gen, _, _ = vq_get_generator(args, device, train_loader, 0, args.data.train_steps)
            b, _ = next(data_gen) 
            b = b.to(device)
            idx = random.randint(0, args.data.batch_size - 1)
            x = b[idx]

            if x.size(0) < n_frames:
                raise ValueError("The sampled video does not have enough frames.")

            max_start_frame = x.size(0) - n_frames
            start_frame = random.randint(0, max_start_frame)
            x_frames = x[start_frame:start_frame + n_frames]
            return x_frames

        def compute_noise_levels(flow_matching, args, dataset_frames, unmasking_steps):
            scheduling_matrix = flow_matching._generate_scheduling_matrix(args.dynamic.scheduling_matrix, dataset_frames)
            scheduling_matrix = (unmasking_steps - scheduling_matrix) / (unmasking_steps + 1e-4)
            return scheduling_matrix

        def rolling_diffusion_sample_fn(sample_size, model, args, device, flow_matching, sampler, input_tensor_type, context_indices=None):
            batch_size, total_length, h, w = sample_size
            total_length = args.dynamic.sampling_horizon
            n_context = args.dynamic.n_context_frames       
            window_size = args.data.video_frames              
            steps_per_frame = args.dynamic.sampling_timesteps
            mask_token_id = args.tokenizer.mask_token_id

            context = sample_frames_from_batch(args, n_context).unsqueeze(0).repeat(batch_size, 1, 1, 1) if context_indices is None else context_indices
            masked_frames = torch.full((batch_size, total_length - n_context, h, w),
                                    mask_token_id, device=device, dtype=torch.long)
            video = torch.cat([context, masked_frames], dim=1)  
            current_window = video[:, 0:window_size].clone()  
            current_t_window = torch.zeros((batch_size, window_size), device=device, dtype=torch.float32)
            if n_context > 0:
                current_t_window[:, :n_context] = 1.0  

            target_init = current_t_window.clone()  
            if n_context < window_size:
                L = window_size - n_context
                for i in range(n_context, window_size):
                    j = i - n_context
                    target_init[:, i] = (L - j - 1) / L

            init_step = 0
            while init_step < steps_per_frame:
                candidate = (init_step + 1) / steps_per_frame  
                effective_t = current_t_window.clone()
                for i in range(n_context, window_size):
                    effective_t[:, i] = min(candidate, target_init[0, i].item())
                
                new_window = sampler.sample_step_with_noise_schedule(
                    xt=current_window.long(),
                    t=effective_t,
                    discretefm=flow_matching,
                    model=model,
                    n_steps=steps_per_frame,
                    model_kwargs={}
                )
    
                for i in range(n_context, window_size):
                    if candidate < target_init[0, i].item():
                        current_window[:, i] = new_window[:, i]
                        current_t_window[:, i] = candidate  
                init_step += 1
            
            video[:, 0:window_size] = current_window

            init_t_window = current_t_window.clone()
            window_start = 0
            while window_start + window_size < total_length:
                current_t_window = init_t_window.clone()
                current_window = video[:, window_start:window_start+window_size].clone()
                while current_t_window[0, n_context].item() < 1.0:
                    current_window = sampler.sample_step_with_noise_schedule(
                        xt=current_window.long(),
                        t=current_t_window,
                        discretefm=flow_matching,
                        model=model,
                        n_steps=steps_per_frame,
                        model_kwargs={}
                    )
                    current_t_window[:, n_context:] = torch.clamp(current_t_window[:, n_context:] + (1/steps_per_frame), max=1.0)
                video[:, window_start:window_start+window_size] = current_window 
                window_start+=1

            current_window = video[:, window_start:]
            current_t_window = target_init.clone()
            leftmost_noisy_index = 0

            while torch.any(current_t_window < 1.0):
                current_window = sampler.sample_step_with_noise_schedule(
                    xt=current_window.long(),
                    t=current_t_window,
                    discretefm=flow_matching,
                    model=model,
                    n_steps=steps_per_frame,
                    model_kwargs={}
                )

                for i in range(window_size):
                    if current_t_window[0, i] < 1.0:
                        current_t_window[:, i] = min(
                            1.0, current_t_window[0, i].item() + (1.0 / steps_per_frame)
                        )

                if abs(current_t_window[0, leftmost_noisy_index].item() - 1.0) < 1e-3:
                    video[:, window_start+leftmost_noisy_index] = current_window[:, leftmost_noisy_index]
                    leftmost_noisy_index += 1
            
            return [video.long()]

        def mgm_sample_fn(unmasking_steps, model, dynamic_n_context_frames, xs_input, args, device, flow_matching, sampler, input_tensor_type, return_chains=False):

            if input_tensor_type == "btwh":
                batch_size, n_frames, w, h = xs_input.shape
            elif input_tensor_type == "btcwh":
                batch_size, n_frames, n_channels, w, h = xs_input.shape
            else:
                raise ValueError(f"Unsupported input_tensor_type: {input_tensor_type}")

            if dynamic_n_context_frames > 0:    
                noise_levels_context = np.full(dynamic_n_context_frames, args.dynamic.stabilization_level, dtype=np.float32)
            else:
                noise_levels_context = np.array([], dtype=np.float32)

            noise_levels = torch.from_numpy(noise_levels_context).to(device).float().unsqueeze(0).repeat(batch_size, 1)

            if args.dynamic.time_cond == 0:
                final_noise_levels = torch.zeros((batch_size, n_frames), device=device)
                final_noise_levels[:, :dynamic_n_context_frames] = noise_levels[:, :dynamic_n_context_frames]
            else:
                final_noise_levels = torch.full((batch_size, n_frames), args.dynamic.time_cond, device=device)
                if dynamic_n_context_frames > 0:
                    final_noise_levels[:, :dynamic_n_context_frames] = noise_levels[:, :dynamic_n_context_frames]
            
            noise_levels = final_noise_levels

            if args.dynamic.partial_context_guidance:
                code_list, mask_list = sampler.sample(
                    sample_size=xs_input.shape,
                    discretefm=flow_matching,
                    model=model,
                    noise_levels=noise_levels,
                    n_steps=unmasking_steps,
                    init_code=xs_input,
                    top_p=0.0,
                    top_k=0,
                    mgm_randomize="linear",
                )

            else:
                code_list, mask_list = sampler.sample_with_partial_context_guidance(
                    sample_size=xs_input.shape,  # (b, t, h, w) or (b, t, c, h, w)
                    discretefm=flow_matching,
                    model=model,
                    noise_levels=noise_levels,
                    n_steps=unmasking_steps,
                    n_context=dynamic_n_context_frames,
                    partial_context_guidance_level=args.dynamic.partial_context_guidance_level,
                    partial_context_guidance_steps=args.dynamic.partial_context_guidance_steps,
                    init_code=xs_input,
                )

            xs_input = code_list[-1]
            return (xs_input, code_list) if return_chains else xs_input

        def unmask_chunk(xs_input, model, dynamic_n_context_frames, return_chains=False, chunk_idx=0, total_chunks=1):
            unmasking_steps = args.dynamic.sampling_timesteps
            dataset_frames = args.data.video_frames
            batch_size = xs_input.shape[0]
        
            if args.dynamic.sampler == "mgm":
                return mgm_sample_fn(
                    unmasking_steps,
                    model,
                    dynamic_n_context_frames,
                    xs_input,
                    args,
                    device,
                    flow_matching,
                    sampler,
                    input_tensor_type,
                    return_chains,
                )
            else:
                return fm_sample_fn(
                    unmasking_steps,
                    model,
                    dynamic_n_context_frames,
                    xs_input,
                    args,
                    device,
                    flow_matching,
                    sampler,
                    input_tensor_type,
                    return_chains
                )

        def fm_sample_fn(unmasking_steps, model, dynamic_n_context_frames, xs_input, args, device, flow_matching, sampler, input_tensor_type, return_chains=False):
            partial_context_guidance_level=args.dynamic.partial_context_guidance_level
            dataset_frames=args.data.video_frames
            batch_size=xs_input.shape[0]
            sampling_chain = None

            if args.dynamic.sampler == "diffusion_forcing":
                noise_schedule="pyramid"
            else:
                noise_schedule="full_sequence"

            scheduling_matrix = compute_noise_levels(flow_matching, args, dataset_frames, unmasking_steps)
            for m in range(scheduling_matrix.shape[0] - 1):
                if dynamic_n_context_frames > 0:    
                    noise_levels_context = np.full(dynamic_n_context_frames, args.dynamic.stabilization_level, dtype=np.float32)
                else:
                    noise_levels_context = np.array([], dtype=np.float32)

                noise_levels_remaining = scheduling_matrix[m][dynamic_n_context_frames:]
                noise_levels = np.concatenate((noise_levels_context, noise_levels_remaining))
                noise_levels = torch.from_numpy(noise_levels).to(device).float().unsqueeze(0).repeat(batch_size, 1)

                if args.dynamic.partial_context_guidance:

                    xs_input = sampler.sample_step_with_partial_context_guidance(
                        xt=xs_input.long(),
                        t=noise_levels,
                        discretefm=flow_matching,
                        model=model,
                        n_steps=unmasking_steps,
                        n_context=dynamic_n_context_frames,
                        partial_context_guidance=args.dynamic.partial_context_guidance,
                        partial_context_guidance_level=args.dynamic.partial_context_guidance_level,
                    )
                else:
                    xs_input = sampler.sample_step_with_noise_schedule(
                        xt=xs_input.long(),
                        t=noise_levels,
                        discretefm=flow_matching,
                        model=model,
                        n_steps=unmasking_steps,
                    )
                    
                if sampling_chain is None:
                    sampling_chain = xs_input.unsqueeze(0)
                else:
                    sampling_chain = torch.cat((sampling_chain, xs_input.unsqueeze(0)), dim=0)
               
            return (xs_input, sampling_chain) if return_chains else xs_input

        def sample_fn(sample_size, model, context_indices=None, return_chains=False, **model_kwargs):
            unmasking_steps = args.dynamic.sampling_timesteps
            dataset_frames = args.data.video_frames  
            batch_size, n_frames, h, w = sample_size
            frames_to_sample = args.dynamic.sampling_horizon  
            n_context_frames = args.dynamic.n_context_frames  
            chunk_size = dataset_frames - n_context_frames  
            sampling_window_stride = args.dynamic.sampling_window_stride  
            mask_token_id = args.tokenizer.mask_token_id
            total_chunks = round((frames_to_sample - dataset_frames) / sampling_window_stride) + 1
            chunk_idx = 0
            chains = []
            assert sampling_window_stride <= chunk_size

            if args.dynamic.sampler == "rolling_diffusion":
                return rolling_diffusion_sample_fn(sample_size, model, args, device, flow_matching, sampler, input_tensor_type, context_indices)

            if n_context_frames == 0:
                horizon = min(frames_to_sample, dataset_frames)
                xs_input = torch.ones((batch_size, dataset_frames, h, w), device=device) * mask_token_id
                xs_input = unmask_chunk(xs_input, model, 0)

                if horizon < dataset_frames:
                    xs_input = xs_input[:, :horizon]

                samples = [xs_input.long()]
                return samples

            if context_indices is not None:
                xs_pred = context_indices
            else:
                x_context_frames = sample_frames_from_batch(args, n_context_frames)
                xs_pred = x_context_frames.unsqueeze(0).repeat(batch_size, 1, 1, 1)  

            curr_frame = n_context_frames

            while curr_frame < frames_to_sample:
                remaining = frames_to_sample - curr_frame

                if remaining <= sampling_window_stride:
                    horizon = remaining
                    dynamic_n_context_frames = dataset_frames - horizon
                    dynamic_n_context_frames = min(dynamic_n_context_frames, xs_pred.shape[1])
                    chunk_size = dataset_frames - dynamic_n_context_frames
                else:
                    horizon = sampling_window_stride
                    dynamic_n_context_frames = n_context_frames

                if dynamic_n_context_frames > 0:
                    xs_context = xs_pred[:, -dynamic_n_context_frames:]  
                else:
                    xs_context = torch.empty((batch_size, 0, h, w), device=device)

                chunk = torch.ones((batch_size, chunk_size, h, w), device=device) * mask_token_id
                xs_input = torch.cat([xs_context, chunk], dim=1)
                
                if return_chains:
                    xs_input, chain = unmask_chunk(xs_input, model, dynamic_n_context_frames, return_chains, chunk_idx, total_chunks)
                    chains.append(chain)
                else:
                    xs_input = unmask_chunk(xs_input, model, dynamic_n_context_frames, False, chunk_idx, total_chunks)

                xs_new = xs_input[:, dynamic_n_context_frames:dynamic_n_context_frames + sampling_window_stride]
                xs_pred = torch.cat([xs_pred, xs_new], dim=1) if xs_pred.shape[1] != 0 else xs_new
                chunk_idx += 1

                curr_frame += horizon

            samples = [xs_pred.long()]
            return (samples, chains) if return_chains else samples
    else:
        raise ValueError(f"dynamic={args.dynamic} not supported")

    return training_losses_fn, sample_fn


def vq_get_encoder_decoder(args, device):
    if args.tokenizer.name in ["titok_s128", "titok_l32"]:
        from titok_1d_tokenizer.modeling.titok import TiTok

        vocab_size = args.tokenizer.vocab_size
        if args.tokenizer.name == "titok_s128":
            _tokenizer = TiTok.from_pretrained(
                "yucornetto/tokenizer_titok_s128_imagenet"
            )
        elif args.tokenizer.name == "titok_l32":
            _tokenizer = TiTok.from_pretrained(
                "yucornetto/tokenizer_titok_l32_imagenet"
            )
        else:
            raise ValueError(f"tokenizer={args.tokenizer.name} not supported")
        _tokenizer.eval()
        _tokenizer.requires_grad_(False)
        _tokenizer = _tokenizer.to(device)

        @torch.no_grad()
        def tokenizer_encode_fn(img):
            img = img / 255.0
            x = _tokenizer.encode(img)[1]["min_encoding_indices"]
            x = x.squeeze(1)
            return x

        @torch.no_grad()
        def tokenizer_decode_fn(indices, mini_bs=25):
            indices[indices == args.tokenizer.mask_token_id] = (
                args.tokenizer.mask_token_reindex
            )
            indices = indices.unsqueeze(1)
            for i in range(0, len(indices), mini_bs):
                _indices = indices[i : i + mini_bs]
                _img = _tokenizer.decode_tokens(_indices)
                if i == 0:
                    img = _img
                else:
                    img = torch.cat([img, _img], dim=0)
            img = torch.clamp(img, 0.0, 1.0)
            img = (img * 255.0).to(dtype=torch.uint8)
            return img

    elif args.tokenizer.name in ["sd_vq_f8", "sd_vq_f8_size512"]:
        use_id = args.input_tensor_type == "bt"
        vocab_size = args.tokenizer.vocab_size
        latent_size = args.tokenizer.latent_size
        config_path = args.tokenizer.config_path
        ckpt_path = args.tokenizer.ckpt_path

        sys.path.insert(0, os.path.abspath("./ldm"))
        from ldm.ldm.util import instantiate_from_config

        config = OmegaConf.load(config_path)
        pl_sd = torch.load(ckpt_path, map_location="cpu")
        sd = pl_sd["state_dict"]
        _tokenizer = instantiate_from_config(config.model)
        _tokenizer.load_state_dict(sd, strict=False)
        _tokenizer.eval()
        _tokenizer.requires_grad_(False)
        _tokenizer = _tokenizer.to(device)

        @torch.no_grad()
        def tokenizer_encode_fn(img, mini_bs=25):
            img = img / 255.0
            img = (img - 0.5) * 2
            # somelogic about video
            img_shape = img.shape
            if len(img_shape) == 5:
                b, t, c, h, w = img.shape
                img = rearrange(img, "b t c h w -> (b t) c h w")
            ############################################################
            for i in range(0, len(img), mini_bs):
                _img = img[i : i + mini_bs]
                encode_res = _tokenizer.encode(_img)
                quant = encode_res[0]
                diff = encode_res[1]
                _indices = encode_res[2][-1]
                if i == 0:
                    indices = _indices
                else:
                    indices = torch.cat([indices, _indices], dim=0)
            ############################################################
            if len(img_shape) == 5:
                indices = rearrange(
                    indices,
                    "(b t h w) -> b t h w",
                    b=b,
                    t=t,
                    h=latent_size,
                    w=latent_size,
                )
            elif len(img_shape) == 4:
                indices = rearrange(
                    indices,
                    "(b h w) -> b h w",
                    b=img_shape[0],
                    h=latent_size,
                    w=latent_size,
                )
            else:
                raise ValueError(f"Unsupported batch dimensions: {len(img_shape)}")
            if use_id:
                raise NotImplementedError
            return indices
            ############################################################

        @torch.no_grad()
        def tokenizer_decode_fn(indices, mini_bs=25):
            indices[indices == args.tokenizer.mask_token_id] = (
                args.tokenizer.mask_token_reindex
            )
            indices_shape = indices.shape
            if len(indices_shape) == 4:  # video
                b, t, h, w = indices.shape
                indices = rearrange(indices, "b t h w -> (b t) (h w)")
            elif len(indices_shape) == 3:  # image
                indices = rearrange(indices, "b h w -> b (h w)")
            else:
                raise ValueError(f"Unsupported batch dimensions: {len(indices_shape)}")
            # somelogic about video

            for i in range(0, len(indices), mini_bs):
                _indices = indices[i : i + mini_bs]
                _img = _tokenizer.decode_tokens(_indices.long())
                if i == 0:
                    img = _img
                else:
                    img = torch.cat([img, _img], dim=0)
            # somelogic about video
            if len(indices_shape) == 4:  # if video
                img = rearrange(img, "(b t) c h w -> b t c h w", b=b, t=t)

            img = img.clamp(-1, 1)
            img = ((img + 1) * 0.5 * 255.0).to(dtype=torch.uint8)
            return img
    else:
        raise ValueError(f"tokenizer={args.tokenizer.name} not supported")

    if "indice" in args.data.name:
        tokenizer_encode_fn = lambda x: x
    return tokenizer_encode_fn, tokenizer_decode_fn


def vq_get_generator(cfg, device, loader, rank_id, train_steps, vae=None):

    def get_data_generator(return_cls_id=True):
        _init = train_steps
        while True:
            for data in tqdm(
                loader,
                disable=rank_id > 0,
                initial=_init,
                desc="data fetching",
            ):
                x = data["image"].to(device)
                try:
                    y = data["cls_id"].to(device)
                except:
                    try:
                        y = data["caption_feat"].to(device)
                    except:
                        y = None
                x = out2img(x)

                if return_cls_id:
                    yield x, y
                else:
                    yield x

    def get_caption_generator():
        while True:
            for data in tqdm(
                loader,
                disable=rank_id > 0,
                desc="gen caption",
            ):
                caption_feat = data["caption_feat"].to(device)
                caption = data["caption"]

                yield caption_feat, caption

    def get_indices_generator(return_cls_id=True):
        _init = train_steps
        while True:
            for data in tqdm(
                loader,
                disable=rank_id > 0,
                initial=_init,
                desc="data fetching",
            ):

                x = data["indices"].to(device)
                try:
                    y = data["cls_id"].to(device)
                except:
                    try:
                        y = data["caption_feat"].to(device)
                    except:
                        y = None

                if return_cls_id:
                    yield x, y
                else:
                    yield x

    if "indices" in cfg.data.name:
        data_gen = get_indices_generator(return_cls_id=True)
        realimg_gen = get_indices_generator(return_cls_id=False)
    else:
        data_gen = get_data_generator(return_cls_id=True)
        realimg_gen = get_data_generator(return_cls_id=False)
    cap_gen = get_caption_generator()
    return data_gen, realimg_gen, cap_gen