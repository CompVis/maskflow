def get_dataloader(args):
    args_data = args.data
    if (
        hasattr(args_data, "video_frames")
        and args_data.video_frames > 0
        and "indices" in args_data.name
    ):
        from datasets_wds.indices_h5_dataloader_video import (
            get_inf_h5_dataloader,
        )

        loader = get_inf_h5_dataloader(**args_data)
        return loader
    elif hasattr(args_data, "video_frames") and args_data.video_frames > 0:
        from datasets_wds.h5_dataloader_video import (
            get_inf_h5_dataloader,
        )

        loader = get_inf_h5_dataloader(**args_data)
        return loader

    elif args_data.name in [
        "scratch_imagenet512_cond",
        "scratch_imagenet256_cond",
        "scratchv2_imagenet256_uncond",
        "scratchv2_imagenet256_cond",
        "scratchv2_imagenet512_cond",
    ]:
        from datasets_wds.web_dataloader_rawbyte import SimpleImageDataset

        datamod = SimpleImageDataset(**args_data)
        if args_data.subset == "train":
            loader = datamod.train_dataloader()
        elif args_data.subset == "val":
            loader = datamod.eval_dataloader()
        else:
            raise ValueError(f"subset {args_data.subset} not supported")
        return loader
    elif "indices" in args_data.name and args_data.name.startswith("coco"):
        from datasets_wds.indices_web_dataloader_t2i import SimpleImageDataset_T2I

        datamod = SimpleImageDataset_T2I(**args_data)
        if args_data.subset == "train":
            loader = datamod.train_dataloader()
        elif args_data.subset == "val":
            loader = datamod.eval_dataloader()
        else:
            raise ValueError(f"subset {args_data.subset} not supported")
        return loader
    elif "indices" in args_data.name and "imagenet" in args_data.name:
        from datasets_wds.indices_web_dataloader_rawbyte import SimpleImageDataset

        datamod = SimpleImageDataset(**args_data)
        if args_data.subset == "train":
            loader = datamod.train_dataloader()
        elif args_data.subset == "val":
            loader = datamod.eval_dataloader()
        else:
            raise ValueError(f"subset {args_data.subset} not supported")

        return loader
    elif args_data.name.startswith("coco"):
        from datasets_wds.web_dataloader_t2i import SimpleImageDataset_T2I

        datamod = SimpleImageDataset_T2I(**args_data)
        if args_data.subset == "train":
            loader = datamod.train_dataloader()
        elif args_data.subset == "val":
            loader = datamod.eval_dataloader()
        else:
            raise ValueError(f"subset {args_data.subset} not supported")
        return loader
    elif args_data.name.startswith("cifar10"):
        from datasets_wds.web_dataloader_v2 import SimpleImageDataset

        datamod = SimpleImageDataset(**args_data)
        if args_data.subset == "train":
            loader = datamod.train_dataloader()
        elif args_data.subset == "val":
            loader = datamod.eval_dataloader()
        else:
            raise ValueError(f"subset {args_data.subset} not supported")
        return loader
    elif args_data.name.startswith("cs"):
        from datasets_wds.h5_dataloader_cs import get_inf_h5_dataloader

        loader = get_inf_h5_dataloader(**args_data)
        return loader
    else:
        raise NotImplementedError(f"data {args_data.name} not supported")
