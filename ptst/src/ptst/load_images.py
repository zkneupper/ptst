
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

def image_loader(image_file_path, imsize=512, device=None):
    """
    # desired size of the output image
    imsize = 512 # int(128 * 4)
    """
    loader = transforms.Compose([
        transforms.Resize(imsize),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor    
    
    image = Image.open(image_file_path)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


# unloader = transforms.ToPILImage()  # reconvert into PIL image

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    unloader = transforms.ToPILImage()  # reconvert into PIL image
    image = unloader(image)
    plt.ion()
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated