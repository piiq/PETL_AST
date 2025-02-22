import os
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import AutoFeatureExtractor


class AsthmaDataset(Dataset):
    """
    Dataset for asthma/not-asthma classification using ESC-50 format.
    The dataset consists of 5-second-long recordings organized into 2 classes,
    arranged in 5 folds for cross-validation.
    """

    def __init__(
        self,
        data_path,
        max_len_AST,
        split,
        train_fold_nums=[1, 2, 3],
        valid_fold_nums=[4],
        test_fold_nums=[5],
        apply_SpecAug=False,
        few_shot=False,
        samples_per_class=1,
    ):
        if split not in ("train", "valid", "test"):
            raise ValueError(f"`split` arg ({split}) must be train/valid/test.")

        self.data_path = os.path.expanduser(data_path)
        self.max_len_AST = max_len_AST
        self.split = split
        self.train_fold_nums = train_fold_nums
        self.valid_fold_nums = valid_fold_nums
        self.test_fold_nums = test_fold_nums

        # SpecAugment parameters (same as ESC-50)
        self.apply_SpecAug = apply_SpecAug
        self.freq_mask = 24
        self.time_mask = 80

        self.x, self.y = self.get_data()

        if few_shot:
            self.x, self.y = self.get_few_shot_data(samples_per_class)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        # Convert numpy array to tensor if not already
        x = (
            torch.from_numpy(self.x[index])
            if isinstance(self.x[index], np.ndarray)
            else self.x[index]
        )
        y = torch.tensor(self.y[index])

        if self.apply_SpecAug:
            frequency_mask = torchaudio.transforms.FrequencyMasking(self.freq_mask)
            time_mask = torchaudio.transforms.TimeMasking(self.time_mask)

            f_bank = torch.transpose(x, 0, 1)
            f_bank = f_bank.unsqueeze(0)
            f_bank = frequency_mask(f_bank)
            f_bank = time_mask(f_bank)
            f_bank = f_bank.squeeze(0)
            f_bank = torch.transpose(f_bank, 0, 1)

            return f_bank, y
        else:
            return x, y

    def get_few_shot_data(self, samples_per_class: int):
        x_few, y_few = [], []
        total_classes = np.unique(self.y)

        for class_ in total_classes:
            cap = 0
            for index in range(len(self.y)):
                if self.y[index] == class_:
                    x_few.append(self.x[index])
                    y_few.append(self.y[index])
                    cap += 1
                    if cap == samples_per_class:
                        break

        return x_few, y_few

    def get_data(self):
        if self.split == "train":
            fold = self.train_fold_nums
        elif self.split == "valid":
            fold = self.valid_fold_nums
        else:
            fold = self.test_fold_nums

        print(f"\nLoading {self.split} data with folds: {fold}")

        processor = AutoFeatureExtractor.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593", max_length=self.max_len_AST
        )

        x, y = [], []
        fold_counts = {}

        # Read our metadata CSV that was created during data preparation
        with open(os.path.join(self.data_path, "meta", "metadata.csv")) as f:
            lines = f.readlines()[1:]  # Skip header

        for line in lines:
            filename, fold_num, target, category, *_ = line.strip().split(",")
            fold_num = int(fold_num)

            # Track counts for all folds
            fold_counts[fold_num] = fold_counts.get(fold_num, 0) + 1

            # Skip if not in current fold
            if fold_num not in fold:
                continue

            # Load and process audio
            audio_path = os.path.join(self.data_path, "audio", filename)
            waveform, _ = torchaudio.load(audio_path)

            # Our clips are already at 16kHz from preparation
            features = processor(
                waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt"
            )["input_values"].squeeze(0)

            x.append(features)
            y.append(int(target) - 1)  # Convert from 1,2 to 0,1 for training

        print(f"Total samples per fold: {fold_counts}")
        print(f"Loaded {len(x)} samples for {self.split} set")

        if len(x) == 0:
            raise ValueError(
                f"No samples found for {self.split} set with folds {fold}. Check your fold assignments in metadata.csv"
            )

        return x, np.array(y)

    @property
    def class_ids(self):
        return {
            "not_asthma": 0,
            "asthma": 1,
        }
