from PIL import Image 
from torch.utils.data import Dataset

class LungDataset(Dataset):
    def __init__(self, img_mask_list, img_folder, mask_folder, transform = (None,None)): # 'Initialization'
        self.img_mask_list = img_mask_list
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.transform = transform
  
    def __len__(self):  # 'Denotes the total number of samples'
        return len(self.img_mask_list)

    def __getitem__(self,index): # 'Generates one sample of data'      

        images_names, masks_names = self.img_mask_list[index]
        images = Image.open(self.img_folder +  images_names).convert('L')# grey
        masks = Image.open(self.mask_folder + masks_names).convert('L') # binary

        if self.transform != (None, None):
            images = self.transform[0](images)
            masks = self.transform[1](masks)
        # masks = masks.long()
        # masks = torch.squeeze(masks)
        return images, masks # chua 1 cap


