import os
import pandas as pd

from torch.utils.data import Dataset

class BHMDataset(Dataset):
    def __init__(self, base_folder, split):
        self.data_set = []

        dataset_path = os.path.join(base_folder, 'Files', split + '_task1_translated_indictrans2.xlsx')

        data = pd.read_excel(dataset_path)

        for _, row in data.iterrows():
            label = 1 if row['Labels'] == 'hate' else 0

            data_entry = {
                "img": os.path.join(base_folder, "Memes", row['image_name']),
                "text": row['Captions_English'],
                "label": label  # Hate (1) or Non-hate (0)
            }
            self.data_set.append(data_entry)

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, index):
        return self.data_set[index]