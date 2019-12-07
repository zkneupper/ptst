import torch
from .load_images import image_loader, imshow


class StyleTransferer:
    def __init__(self, device=None):

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.content_img = None
        self.style_img = None

    def __repr__(self):
    	return self.__class__.__name__
