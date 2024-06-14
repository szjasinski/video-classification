import pathlib

from utilities.data_loader_utilities import download_ufc_101_subset
from utilities.data_loader_utilities import get_batched_train_val_test


class DataLoader:
    def __init__(self, num_classes, n_frames, data_split, batch_size):
        self.URL = 'https://storage.googleapis.com/thumos14_files/UCF101_videos.zip'
        self.download_dir = pathlib.Path('./UCF101_subset/')

        self.num_classes = num_classes
        self.n_frames = n_frames
        self.data_split = data_split
        self.batch_size = batch_size

    def load(self):
        subset_paths = download_ufc_101_subset(self.URL,
                                               num_classes=self.num_classes,
                                               splits={"train": self.data_split['train'],
                                                       "val": self.data_split['val'],
                                                       "test": self.data_split['test']},
                                               download_dir=self.download_dir)

        train_ds, val_ds, test_ds = get_batched_train_val_test(subset_paths=subset_paths, n_frames=self.n_frames,
                                                               batch_size=self.batch_size)

        return train_ds, val_ds, test_ds
