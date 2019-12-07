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

    def set_content_img(self, image_file_path, imsize=512):
        """
        ADD PEP8 DOCSTRING HERE.
        """
        self.content_img = image_loader(
            image_file_path, imsize=imsize, device=self.device
        )

    def set_style_img(self, image_file_path, imsize=512):
        """
        ADD PEP8 DOCSTRING HERE.
        """
        self.style_img = image_loader(
            image_file_path, imsize=imsize, device=self.device
        )

    def check_image_size_match(self):
        """
        ADD PEP8 DOCSTRING HERE.

        We need to style and content images of the same size.
        """
        return self.style_img.size() == self.content_img.size()


# cnn = models.vgg19(pretrained=True).features.to(device).eval()
# cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
# cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
