import torch
from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import make_dataset
from data_set.video_utils import VideoClips, _collate_fn
from torchvision.datasets.vision import VisionDataset
from torchvision.io import read_video_timestamps, read_video
from tqdm import tqdm
from typing import List, Tuple, Optional
from random import randint

class HandWashDataset(VisionDataset):
    """
    Args:
        root (string): Root directory of the HandWash Dataset.
        frames_per_clip (int): number of frames in a clip
        step_between_clips (int): number of frames between each clip
        transform (callable, optional): A function/transform that  takes in a TxHxWxC video
            and returns a transformed version.

    Returns:
        video (Tensor[T, H, W, C]): the `T` video frames
        label (int): class of the video clip
    """

    def __init__(self, root, frames_per_clip, step_between_clips=1, frame_rate=None,
                 extensions=('mp4',), transform=None, _precomputed_metadata=None,
                 num_workers=8, _video_width=0, _video_height=0,
                 _video_min_dimension=0, _audio_samples=0, _audio_channels=0):
        super(HandWashDataset, self).__init__(root)

        classes = list(sorted(list_dir(root)))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file=None)
        self.classes = classes
        self.frame_per_clip = frames_per_clip
        video_list = [x[0] for x in self.samples]
        self.video_clips = VideoClips(
            video_list,
            frames_per_clip,
            step_between_clips,
            frame_rate,
            _precomputed_metadata,
            num_workers=num_workers,
            _video_width=_video_width,
            _video_height=_video_height,
            _video_min_dimension=_video_min_dimension,
            _audio_samples=_audio_samples,
            _audio_channels=_audio_channels,
        )
        self.transform = transform

    @property
    def metadata(self):
        return self.video_clips.metadata

    def __len__(self):
        return self.video_clips.num_clips()

    def __getitem__(self, idx):
        video, audio, info, video_idx = self.video_clips.get_clip(idx)

        label = self.samples[video_idx][1]
        
        if self.transform is not None:
            video = self.transform(video)

        return video, label

class _VideoFrameLengthDataset:
    """
    Dataset used to parallelize the reading of the video frame length
    of a list of videos, given their paths in the filesystem.
    """

    def __init__(self, video_paths: List[str]) -> None:
        self.video_paths = video_paths

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> Tuple[List[int], Optional[float]]:
        return self.video_paths[idx], len(read_video_timestamps(self.video_paths[idx])[0])

class OneSamplePerVideoDataset(VisionDataset):
    """
    Args:
        root (string): Root directory of the HandWash Dataset.
        frames_per_clip (int): number of frames in a clip
        transform (callable, optional): A function/transform that  takes in a TxHxWxC video
            and returns a transformed version.

    Returns:
        video (Tensor[T, H, W, C]): the `T` video frames
        label (int): class of the video clip
    """

    def __init__(self, root, frames_per_clip, extensions=('mp4',), transform=None):
        super(OneSamplePerVideoDataset, self).__init__(root)

        classes = list(sorted(list_dir(root)))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file=None)
        self.classes = classes
        self.frame_per_clip = frames_per_clip
        self.video_path = [x[0] for x in self.samples]
        self.transform = transform
        self.video_list = []
        self.remove_less_frame_video()

    def remove_less_frame_video(self):
        video_label_dict = {x[0] : x[1] for x in self.samples}
        dl: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            _VideoFrameLengthDataset(self.video_path),  # type: ignore[arg-type]
            batch_size=32,
            num_workers=8,
            collate_fn=_collate_fn,
        )

        with tqdm(total=len(dl)) as pbar:
            for batch in dl:
                pbar.update(1)
                video_paths, frame_lengths = list(zip(*batch))
                for video_path, frame_length in zip(video_paths, frame_lengths):
                    if frame_length >= 8:
                        self.video_list.append((video_path, video_label_dict[video_path]))
    
    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        video_name , label = self.video_list[idx]
        video, audio, info = read_video(video_name)

        start_frame = randint(0, video.shape[0] - self.frame_per_clip)
        end_frame = start_frame + self.frame_per_clip - 1
        video = video[start_frame:end_frame]
        
        if self.transform is not None:
            video = self.transform(video)

        return video, label