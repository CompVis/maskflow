import h5py
import numpy as np

original_h5_path = "/dss/dsshome1/02/di38taq/droll/data/faceforensics_train_indices.h5"
new_h5_path = "/dss/dsshome1/02/di38taq/droll/data/faceforensics_train_indices_32.h5"

new_clip_length = 32
frame_interval = 3


with h5py.File(original_h5_path, "r") as original_h5:
    video_data = original_h5["video"][:]  
    original_start_index_list = original_h5["start_index_list"][:]

    new_start_index_list = []

    for idx, (start, end) in enumerate(original_start_index_list):
        total_frames = end - start
        print(f"Video {idx}: Total frames = {total_frames}")

        for new_start in range(start, end - new_clip_length + 1, frame_interval):
            new_end = new_start + new_clip_length
            if new_end <= end:
                new_start_index_list.append([new_start, new_end])

    new_start_index_list = np.array(new_start_index_list)

    with h5py.File(new_h5_path, "w") as new_h5:
        new_h5.create_dataset("video", data=video_data, dtype=np.int32)
        new_h5.create_dataset("start_index_list", data=new_start_index_list)

    print(f"Saved new .h5 file with updated indices to {new_h5_path}.")
