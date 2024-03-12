import random
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
class LengthBatchSampler:
    def __init__(self, lengths: torch.Tensor, max_len, ignore_len, shuffle=False):
        # super().__init__(data_source)

        self.max_len = max_len
        self.ignore_len = ignore_len

        self.shuffle = shuffle
        self.lengths = lengths
        self.batches = self._create_batches()

    def _create_batches(self):

        sorted = torch.sort(self.lengths).values
        filtered_indices = torch.arange(0, len(sorted), dtype=torch.long, device=self.lengths.device)[
            sorted <= self.ignore_len]

        batches = []
        current_batch = []
        max_len = 0
        current_n = 0

        for i in tqdm(filtered_indices, desc="creating batches", total=len(filtered_indices)):
            item_len = sorted[i]
            max_len = max(max_len, item_len)
            current_n+=1
            if (current_n+1) * max_len < self.max_len:
                current_batch.append(i.item())

            else:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [i.item()]
                max_len = 0
                current_n = 0


        if current_batch:
            batches.append(current_batch)
        return batches

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)