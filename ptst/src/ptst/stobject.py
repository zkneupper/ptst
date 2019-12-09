import torch
import torchvision.models as models
from .load_images import image_loader, imshow
from .styletransfer import run_style_transfer

class StyleTransferer:
    def __init__(self, device=None):

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.content_img = None
        self.style_img = None
        self.input_img = None
        self.output_img = None
        self._cnn = None
        self._cnn_normalization_mean = None
        self._cnn_normalization_std = None
        self._content_layers_default = None
        self._style_layers_default = None


    def __repr__(self):
        return self.__class__.__name__

    def set_content_img(self, image_file_path, imsize=512):
        """
        ADD PEP8 DOCSTRING HERE.
        """
        self.content_img = image_loader(
            image_file_path, imsize=imsize, device=self.device
        )

    def set_input_img(self):
        """
        EDIT THIS FUNCTION!
        ADD PEP8 DOCSTRING HERE.
        """
        # add the original input image to the figure:
        self.input_img = self.content_img.clone()
        # if you want to use white noise instead uncomment the below line:
        # self.input_img = torch.randn(content_img.data.size(), device=device)


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

    def initialize_cnn(self):
        """
        ADD PEP8 DOCSTRING HERE.
        """
        self._cnn = models.vgg19(pretrained=True).features.to(self.device).eval()
        self._cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self._cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
        self._content_layers_default = ['conv_4']
        self._style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


    def execute_style_transfer(self,
        num_steps=300,
        style_weight=1000000,
        content_weight=1,
        file_path_output=None,
        verbose=1,
        ):
        """
        ADD PEP8 DOCSTRING HERE.
        """
        assert self._cnn is not None
        assert self._cnn_normalization_mean is not None
        assert self._cnn_normalization_std is not None
        assert self._content_layers_default is not None
        assert self._style_layers_default is not None

        self.output_img = run_style_transfer(
            self._cnn,
            self._cnn_normalization_mean,
            self._cnn_normalization_std,
            self.content_img,
            self.style_img,
            self.input_img,
            num_steps=num_steps,
            style_weight=style_weight,
            content_weight=content_weight,
            file_path_output=file_path_output,
            verbose=verbose,
            )


# def get_input_optimizer(input_img):
#     # this line to show that input is a parameter that requires a gradient
#     optimizer = optim.LBFGS([input_img.requires_grad_()])
#     return optimizer
