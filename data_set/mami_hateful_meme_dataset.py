import csv
import os

from torch.utils.data import Dataset


class MAMIDataset(Dataset):
    def __init__(self, base_folder, split):
        self.data_set = []

        dataset_path = os.path.join(base_folder, split + '.tsv')

        # Open and parse the TSV file, directly populating self.data_set
        with open(dataset_path, 'r') as tsv_file:
            reader = csv.DictReader(tsv_file, delimiter='\t')

            for row in reader:
                data_entry = {
                    "img": "images/" + row['file_name'],
                    "label": int(row['label']),
                    "text": row['text']
                }
                self.data_set.append(data_entry)

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, index):
        return self.data_set[index]
