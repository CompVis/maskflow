import torch
import torchvision.transforms as transforms
import wids

import wandb
from wandb_utils import array2grid_pixel

cifar10_classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
cityscapes_big8 = [
    "Flat",
    "Human",
    "Vehicle",
    "Construction",
    "Object",
    "Nature",
    "Sky",
    "Void",
]


import torch.distributed as dist

import logging


def print_rank_0(*args, **kwargs):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args, **kwargs)
            # logging.info(*args, **kwargs)
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


def get_version_number():
    # "klv1"  #
    # "klv2"  # update vocab size bug, should add 1 to vocab size to consider the mask token
    # "klv2.1"  # kl cifar10 training bugfix, data range should be [-1,1]
    # "klv3"  # DiT dynamic learn_sigma by default!!!
    # "kl4"  # fix vae encode bug, we multiplied 0.18215 twice!!
    # v4.1 in sampling process, you don't use y at all, which is buggy;decrease sampling step from 1k to 250;raise NotImplementedError;
    # "kl4.2"  # sample_fid_bs = batch_size, which can significantly improve sampling speed
    # "kl4.3"  # sample y uniformly
    # "kl5"  # sample ground truth images uniformly over the dataset
    # "kl5.1"  # use titok's webdataset loader, we extra add horizontal flip; the randomness problem in wds should be gone.
    # "kl5.2"  # remove the epoch in the webdataset
    # "kl5.3"  # add the webdataset from titok always from ImageNet256,512; For previous imagenet256 dataset, random_crop=false,random_flip=false
    ######################
    # "vqv1"  # fix bug that sample_fn, training_losses_fn not use the newest <model>
    # "vqv2"  # update vocab size bug, should add 1 to vocab size to consider the mask token
    # "vq2.1"  # add vq_debug function
    # "vq2.2"  # fix the input token id overlapping with mask_token_id=8, input should be [0,1,2,3,4,5,6,7] other than [0,1,2,3,4,5,6,7,8]; and add extra logs for input.
    # "vq2.3"  # vq_debug less frequently
    # "vq2.4"  # fix campbell_d3pm, campbell sampling bugs
    # "vq3"  # fix a stupid bug, that the logits only consider 255 possibilies in cifar10, while it should consider 256(extra 1 with mask_token), so that your x0 predictior will never predict mask_token again.
    # there should  not so big difference, it only influence the predictor-corrector sampling in discretefm.
    # and in v3, we use default step_num=999 for all dynamics. in previous version,for sampling steps, d3pm=1000,discretefm=100,  campbell=100, campbell_d3pm=100, dynradd=100.
    # "vq3.1"  # fix the bug in ddit2d_b2,ddit2d_s2, we cannot use nn.Embedding and seq_conv together in the very beggining, it seems that it will lead to embedding collapse. Therefore, we choose to simply normlize to [-1,1] in the very beginning.
    # "vq3.2"  # set fid_bs=batch_size, which can significantly improve sampling speed, set discrete sampling step from 1k to 500. This two changes significantly decrease the evaluation time from 6 hours to 2 hours.
    # "vq3.3"  ## sample y uniformly
    # "vq3.4"  ##sample_batch_size=50
    # "vq3.4.1"  ##fix a bug that campbell_d3pm always use 1k sampling steps, which is not flexible.
    # "vq4"  # sample ground truth images uniformly over the dataset
    # "vq4.1"  # use titok's webdataset loader, we extra add horizontal flip; the randomness problem in wds should be gone.
    # "vq4.2"  # remove the epoch in the webdataset
    # return "5.3"  # add the webdataset from titok always from ImageNet256,512,  For previous imagenet256 dataset, random_crop=false,random_flip=false
    # return "5.4"  # coco14 fix bug using real fid statistics, previously it was using fake fid statistics from Imagenet256; previous imagenet256 using web_dataloader_v2, now using web_dataloader_rawbyte always no matter 256 or 512
    # return "5.5"  # fffix a class_num=-1 bug in maskgit, which main influence the training loss convergence; fix the output logit dimenstion bugs, to make class-conditional generation works
    # return "5.5.1"  # add torch.compile
    # return "5.5.2"  # COCO use validate FID statistics
    # return "5.5.3"  # uni_uvit2dv2_label_s2 use **MLP** time embedder, default CFG=4 for UnifiedCFG
    # return "5.6"  # COCO use new FID stats npz_real: ./data/uvit_fid_stats/fid_stats_mscoco256_val.npz
    # return "5.7"  # default use_cfg=false, don't use cfg for sampling in training anh more, it doesn't help as an indicator of the training quality.
    # return "5.8"  # an important update to support gradient accumulation in training, if you set it to 1, it is equivalent to no accumulation. should be bug-free. Also, previous version's gradient clipping is not used, I put it in the right position, but I comment it out.
    return "5.8.1"  # add [X_CFG] tokens, I forgot it previously, so that I cannot do the p(y|x) sampling previously.


def has_label(dataset_name):
    if dataset_name.startswith("ffs"):
        return False
    else:
        return True


def get_dataset_id2label(dataset_name):
    if "imagenet" in dataset_name:
        imagenet_id2realname = open("./datasets_wds/imagenet1k_name.txt").readlines()
        imagenet_id2realname = [
            _cls.strip().split()[-1] for _cls in imagenet_id2realname
        ]
        return imagenet_id2realname
    elif "cifar10" in dataset_name:
        return cifar10_classes
    elif "cs" in dataset_name:
        return cityscapes_big8
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_wid_dl(
    shuffle=True,
    num_workers=1,
    batch_size=4,
    json_path="./data/imagenet256_raw_wds_train.json",
):
    wids_dataset = wids.ShardListDataset(json_path)  # keep=True)

    class _WIDSDataset(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
            self.transform_train = transforms.Compose(
                [
                    # transforms.ToTensor(),
                    transforms.PILToTensor(),
                ]
            )

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            sample = self.dataset[idx]
            _img = sample[".image.jpg"]
            _img = self.transform_train(_img)
            _cls_id = int(sample[".cls_id.cls"])
            return _img, _cls_id  # , _cls_name

    dl = torch.utils.data.DataLoader(
        _WIDSDataset(wids_dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return dl


def get_inf_wid_dl_imglabel(
    args,
    batch_size,
    shuffle=True,
    num_workers=4,
    device=None,
):
    if "imagenet" in args.data.name:
        _dl = get_wid_dl(
            shuffle=shuffle,
            num_workers=num_workers,
            batch_size=batch_size,
            json_path=args.data.tar_base_wid_json,
        )
    elif "cifar10" in args.data.name:
        _dl = get_wid_dl(
            shuffle=shuffle,
            num_workers=num_workers,
            batch_size=batch_size,
            json_path=args.data.tar_base_wid_json,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.data.name}")

    def inifite_generator():
        while True:
            for _img, _label in _dl:
                # image range [0, 255], label start from 0
                yield _img.to(device), _label.to(device)

    return inifite_generator()


def get_inf_wid_dl_imgonly(args, batch_size, device, shuffle=True, num_workers=4):
    gen = get_inf_wid_dl_imglabel(args, batch_size, shuffle, num_workers, device=device)
    for img, cls_id in gen:
        yield img.to(device)


if __name__ == "__main__":
    pass
