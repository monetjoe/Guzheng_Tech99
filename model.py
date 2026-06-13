import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from utils import EN_US


class Interpolate(nn.Module):
    def __init__(
        self,
        size=None,
        scale_factor=None,
        mode="bilinear",
        align_corners=False,
    ):
        super(Interpolate, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )


class EvalNet:
    def __init__(
        self,
        backbone: str,
        cls_num: int,
        ori_T: int,
        imgnet_ver="v1",
        weight_path="",
    ):
        if not hasattr(models, backbone):
            raise ValueError(f"Unsupported model {backbone}.")

        self.imgnet_ver = imgnet_ver
        self.training = bool(weight_path == "")
        self.type, self.weight_url, self.input_size = self._model_info(backbone)
        self.model: torch.nn.Module = eval("models.%s()" % backbone)
        self.ori_T = ori_T
        self.out_channel_before_classifier = 0
        self._set_channel_outsize()  # set out channel size
        self.cls_num = cls_num
        self._set_classifier()
        self._pseudo_foward()
        checkpoint = torch.load(
            weight_path,
            map_location=torch.device("cuda:0") if torch.cuda.is_available() else "cpu",
        )
        if self.type == "squeezenet":
            self.model.load_state_dict(checkpoint, False)

        else:
            self.model.load_state_dict(checkpoint["model"], False)
            self.classifier.load_state_dict(checkpoint["classifier"], False)

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.classifier = self.classifier.cuda()

        self.model.eval()

    def _get_backbone(self, backbone_ver, backbone_list):
        for backbone_info in backbone_list:
            if backbone_ver == backbone_info["ver"]:
                return backbone_info

        raise ValueError("[Backbone not found] Please check if --model is correct!")

    def _model_info(self, backbone: str):
        if EN_US:
            from datasets import load_dataset

            backbone_list = load_dataset("monetjoe/cv_backbones", split="train")

        else:
            from modelscope.msdatasets import MsDataset

            backbone_list = MsDataset.load("monetjoe/cv_backbones", split="train")

        backbone_info = self._get_backbone(backbone, backbone_list)
        return (
            str(backbone_info["type"]),
            str(backbone_info["url"]),
            int(backbone_info["input_size"]),
        )

    def _create_classifier(self):
        original_T_size = self.ori_T
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),  # F -> 1
            nn.ConvTranspose2d(
                self.out_channel_before_classifier,
                256,
                kernel_size=(1, 4),
                stride=(1, 2),
                padding=(0, 1),
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(
                256, 128, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(
                128, 64, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(
                64, 32, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),  # input for Interp: [bsz, C, 1, T]
            Interpolate(
                size=(1, original_T_size), mode="bilinear", align_corners=False
            ),  # classifier
            nn.Conv2d(32, 32, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, self.cls_num, kernel_size=(1, 1)),
        )

    def _set_channel_outsize(self):  #### get the output size before classifier ####
        conv2d_out_ch = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                conv2d_out_ch.append(module.out_channels)

            if (
                str(name).__contains__("classifier")
                or str(name).__eq__("fc")
                or str(name).__contains__("head")
            ):
                if isinstance(module, torch.nn.Conv2d):
                    conv2d_out_ch.append(module.in_channels)
                    break

        self.out_channel_before_classifier = conv2d_out_ch[-1]

    def _set_classifier(self):  #### set custom classifier ####
        if self.type == "resnet":
            self.model.avgpool = nn.Identity()
            self.model.fc = nn.Identity()
            self.classifier = self._create_classifier()

        elif (
            self.type == "vgg" or self.type == "efficientnet" or self.type == "convnext"
        ):
            self.model.avgpool = nn.Identity()
            self.model.classifier = nn.Identity()
            self.classifier = self._create_classifier()

        elif self.type == "squeezenet":
            self.model.classifier = nn.Identity()
            self.classifier = self._create_classifier()

    def get_input_size(self):
        return self.input_size

    def _pseudo_foward(self):
        temp = torch.randn(4, 3, self.input_size, self.input_size)
        out = self.model(temp)
        self.H = int(np.sqrt(out.size(1) / self.out_channel_before_classifier))

    def forward(self, x):
        if torch.cuda.is_available():
            x = x.cuda()

        if self.type == "convnext":
            out = self.model(x)
            return self.classifier(out).squeeze()

        else:
            out = self.model(x)
            out = out.view(
                out.size(0), self.out_channel_before_classifier, self.H, self.H
            )
            return self.classifier(out).squeeze()


class t_EvalNet:
    def __init__(
        self,
        backbone: str,
        cls_num: int,
        ori_T: int,
        imgnet_ver="v1",
        weight_path="",
    ):
        if not hasattr(models, backbone):
            raise ValueError(f"Unsupported model {backbone}.")

        self.imgnet_ver = imgnet_ver
        self.type, self.weight_url, self.input_size = self._model_info(backbone)
        self.model: torch.nn.Module = eval("models.%s()" % backbone)
        self.ori_T = ori_T
        if self.type == "vit":
            self.hidden_dim = self.model.hidden_dim
            self.class_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))

        elif self.type == "swin_transformer":
            self.hidden_dim = 768

        self.cls_num = cls_num
        self._set_classifier()
        checkpoint = torch.load(
            weight_path,
            map_location=torch.device("cuda:0") if torch.cuda.is_available() else "cpu",
        )
        self.model.load_state_dict(checkpoint["model"], False)
        self.classifier.load_state_dict(checkpoint["classifier"], False)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.classifier = self.classifier.cuda()

        self.model.eval()

    def _get_backbone(self, backbone_ver, backbone_list):
        for backbone_info in backbone_list:
            if backbone_ver == backbone_info["ver"]:
                return backbone_info

        raise ValueError("[Backbone not found] Please check if --model is correct!")

    def _model_info(self, backbone: str):
        if EN_US:
            from datasets import load_dataset

            backbone_list = load_dataset("monetjoe/cv_backbones", split="train")

        else:
            from modelscope.msdatasets import MsDataset

            backbone_list = MsDataset.load("monetjoe/cv_backbones", split="v1")

        backbone_info = self._get_backbone(backbone, backbone_list)
        return (
            str(backbone_info["type"]),
            str(backbone_info["url"]),
            int(backbone_info["input_size"]),
        )

    def _create_classifier(self):
        original_T_size = self.ori_T
        self.avgpool = nn.AdaptiveAvgPool2d((1, None))  # F -> 1
        return nn.Sequential(  # nn.AdaptiveAvgPool2d((1, None)), # F -> 1
            nn.ConvTranspose2d(
                self.hidden_dim, 256, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(
                256, 128, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(
                128, 64, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(
                64, 32, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),  # input for Interp: [bsz, C, 1, T]
            Interpolate(
                size=(1, original_T_size), mode="bilinear", align_corners=False
            ),  # classifier
            nn.Conv2d(32, 32, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, self.cls_num, kernel_size=(1, 1)),
        )

    def _set_classifier(self):  #### set custom classifier ####
        if self.type == "vit" or self.type == "swin_transformer":
            self.classifier = self._create_classifier()

    def get_input_size(self):
        return self.input_size

    def forward(self, x: torch.Tensor):
        if torch.cuda.is_available():
            x = x.cuda()
            self.class_token = self.class_token.cuda()

        if self.type == "vit":
            x = self.model._process_input(x)
            batch_class_token = self.class_token.expand(x.size(0), -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            x = self.model.encoder(x)
            x = x[:, 1:].permute(0, 2, 1)
            x = x.unsqueeze(2)
            return self.classifier(x).squeeze()

        elif self.type == "swin_transformer":
            x = self.model.features(x)  # [B, H, W, C]
            x = x.permute(0, 3, 1, 2)
            x = self.avgpool(x)  # [B, C, 1, W]
            return self.classifier(x).squeeze()

        return None
