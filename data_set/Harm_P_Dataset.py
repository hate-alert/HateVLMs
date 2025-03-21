import json
import os

from torch.utils.data import Dataset


class Harm_P_Dataset(Dataset):
    def __init__(self, base_folder, split):
        self.data_set = []

        dataset_path = os.path.join(base_folder, split + '.jsonl')
        with open(dataset_path, 'r') as json_file:
            json_list = list(json_file)

        for json_str in json_list:
            result = json.loads(json_str)
            result['label'] = 0 if 'not harmful' in result['labels'] else 1
            result['img'] = 'images/' + result['image']
            self.data_set.append(result)

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, index):
        return self.data_set[index]
