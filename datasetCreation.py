import os
from PIL import Image
from torch.utils.data import Dataset
import torch.utils.data as data

class EmotionDataset(Dataset):
    def __init__(self, root_dir, mode='train'):
        """Create the dataset object

        Args:
            root_dir (str): absolute path toward the folder containing three folders train, test and validation
                these three folder must contain a folder per category (named like the category) and must contain the augmented images
            mode (str, optional): Mode of the dataset. Can only be "train", "test" or "validation". Defaults to 'train'.

        Raises:
            AttributeError: If the mode is different of "train", "test" or "validation".
            FileNotFoundError: If the directory structure isn't correct
        """
        if mode != "train" and mode != 'test' and mode != "validation":
            raise AttributeError("the mode argument must be either 'train' or 'test' or 'validation'!")

        self.root_dir = root_dir # Directory where we can find train and test folder
        self.categories = {'angry': 0, 'disgusted': 1, 'fearful': 2, 'happy': 3,
                           'neutral': 4, 'sad': 5, 'surprised': 6} # Defines the category in order to keep track of the label
        self.images = [] # store the images path
        self.labels = [] # store the label

        # Determine train or test directory
        data_dir = os.path.join(self.root_dir, mode)
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Directory {data_dir} not found.")

        # Iterate over each category
        for category_name, category_id in self.categories.items():
            category_dir = os.path.join(data_dir, category_name)
            if not os.path.isdir(category_dir):
                raise FileNotFoundError(f"Directory {category_dir} not found.")

            # Iterate over images in the category directory
            for filename in os.listdir(category_dir):
                image_path = os.path.join(category_dir, filename)
                self.images.append(image_path)
                self.labels.append(category_id)

    def __len__(self):
        # return the number of the images depending on the self.mode
        return len(self.images)

    def __getitem__(self, idx):
        # retrieve image and label for idx
        image_path = self.images[idx]
        label = self.labels[idx]

        # open image
        image = Image.open(image_path)

        # if not in gray mode, convert it
        if image.mode != 'L':
            image = image.convert('L')

        return image, label
    

def getDataLoader(data_path, batch_size, num_workers = 0, mode = 'train'):
    """Return the dataloader used for the specified mode

    Args:
        data_path (str): absolute path toward the folder containing the augmented data
        batch_size (int): size to use for the batches
        num_workers (int, optional): nb of process to use for the dataLoader (the more the faster). Defaults to 0.
        mode (str, optional): The mode of the asked dataloader. Defaults to 'train'.

    Returns:
        _type_: _description_
    """
    if mode == 'train':
        shuffle = True
    else:
        shuffle = False
    return data.DataLoader(EmotionDataset(data_path, mode= mode), batch_size=batch_size, shuffle=shuffle, num_workers = num_workers)

