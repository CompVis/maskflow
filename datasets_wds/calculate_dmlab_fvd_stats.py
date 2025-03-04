import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as TF

import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as TF
from einops import rearrange

try:
    from datasets_wds.video_utils import TemporalRandomCrop
except ImportError:
    from video_utils import TemporalRandomCrop

class NPZVideoDataset(Dataset):
    def __init__(self, npz_dir, video_frames=36, frame_interval=1, image_size=256, deterministic=False):
        self.npz_dir = npz_dir
        self.npz_files = [os.path.join(npz_dir, f) for f in os.listdir(npz_dir) if f.endswith('.npz')]
        self.video_frames = video_frames
        self.frame_interval = frame_interval
        self.image_size = image_size
        self.temporal_sample = TemporalRandomCrop(video_frames * frame_interval)
        self.deterministic = deterministic
        if self.deterministic:
            self.rand_state = np.random.RandomState(42)

        self.video_entries = []
        for npz_file in self.npz_files:
            data = np.load(npz_file)
            
            if 'video' in data.files:
                video = data['video']  
                if video.shape[0] >= self.video_frames:
                    self.video_entries.append(npz_file)
            data.close()

        self.transform = TF.Compose([
            TF.Resize((self.image_size, self.image_size)),
            TF.ToTensor(),
        ])

    def __len__(self):
        return len(self.video_entries)

    def __getitem__(self, idx):
        npz_file = self.video_entries[idx]
        data = np.load(npz_file)
        video = data['video']  
        data.close()

        total_frames = video.shape[0]
        
        if self.deterministic:
            start_frame_ind, end_frame_ind = self._deterministic_temporal_sample(total_frames)
        else:
            start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)

        frame_indices = np.linspace(start_frame_ind, end_frame_ind - 1, self.video_frames, dtype=int)
        sampled_frames = video[frame_indices]  
        
        
        transformed_frames = []
        for frame in sampled_frames:
            img = Image.fromarray(frame) 
            img = self.transform(img)  
            transformed_frames.append(img)

        
        transformed_frames = torch.stack(transformed_frames, dim=0)
        return transformed_frames

    def _deterministic_temporal_sample(self, total_frames):
        
        length = self.video_frames * self.frame_interval
        start = self.rand_state.randint(0, total_frames - length + 1)
        end = start + length
        return start, end


def get_npz_video_dataloader(npz_dir, batch_size, num_workers=4, deterministic=False):
    dataset = NPZVideoDataset(npz_dir, deterministic=deterministic)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader


def gen_framefid_stat_video(
    real_img_root,
    dl,
    fid_num=50000,
):
    if not os.path.exists(real_img_root):
        os.makedirs(real_img_root)
    out_path_npy = real_img_root + ".npz"
    idx = 0

    
    for batch_frames in tqdm(dl, desc="Processing frames for FID"):
        
        
        
        B, T, C, H, W = batch_frames.shape
        batch_frames = batch_frames.reshape(B*T, C, H, W)  

        for frame in batch_frames:
            frame = frame.cpu()  
            frame = frame.permute(1, 2, 0).numpy()
            frame = (frame * 255).astype(np.uint8)
            img = Image.fromarray(frame)
            img.save(f"{real_img_root}/{idx}.jpg")
            idx += 1
            if idx >= fid_num:
                break
        if idx >= fid_num:
            break

    from utils.eval_tools.fid_score import calculate_real_statistics

    calculate_real_statistics(
        path_real=real_img_root,
        out_path_npy=out_path_npy,
        device='cuda'  
    )
    print(f"FID statistics saved to {out_path_npy}")


def main():
    
    sys.path.append("/dss/dsshome1/02/di38taq/droll")

    npz_dir = '/dss/dsshome1/02/di38taq/droll/data/dmlab/train'  
    batch_size = 50  
    fid_num = 50000  

    dataloader = get_npz_video_dataloader(npz_dir, batch_size=batch_size, num_workers=4)

    real_img_root = "dmlab_fid_real_images"
    gen_framefid_stat_video(
        real_img_root=real_img_root,
        dl=dataloader,
        fid_num=fid_num,
    )

if __name__ == "__main__":
    main()