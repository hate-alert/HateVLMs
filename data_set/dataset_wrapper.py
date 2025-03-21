from torch.utils.data import ConcatDataset

from data_set.Harm_C_Dataset import Harm_C_Dataset
from data_set.Harm_P_Dataset import Harm_P_Dataset
from data_set.facebook_hateful_meme_dataset import FacebookHatefulMemeDataset
from data_set.mami_hateful_meme_dataset import MAMIDataset
from data_set.BHM_dataset import BHMDataset
from data_set.hinglish_dataset import HinglishDataset


class DatasetWrapper:
    def __init__(self, dataset_tag, base_folder, split):
        match dataset_tag:
            case "facebook_hateful_meme_dataset":
                self.dataset = FacebookHatefulMemeDataset(
                    base_folder=base_folder,
                    split=split
                )
            case "MAMI_dataset":
                self.dataset = MAMIDataset(
                  base_folder=base_folder,
                  split=split
                )
            case "Harm_P_Dataset":
                self.dataset = Harm_P_Dataset(
                    base_folder=base_folder,
                    split=split
                )
            case "Harm_C_Dataset":
                self.dataset = Harm_C_Dataset(
                    base_folder=base_folder,
                    split=split
                )
            case "BHM_Dataset":
                self.dataset = BHMDataset(
                    base_folder=base_folder,
                    split=split
                )
            case "Hinglish_Dataset":
                self.dataset = HinglishDataset(
                    base_folder=base_folder,
                    split=split
                )

    def get_dataset(self):
        return self.dataset
