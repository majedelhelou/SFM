import torch
import torch.nn as nn
import torchvision

class VGGFeatureExtractor(nn.Module):
    def __init__(self, device, feature_layer=34, use_bn=False, use_input_norm=True):
        super(VGGFeatureExtractor, self).__init__()
        self.use_input_norm = use_input_norm
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485 - 1, 0.456 - 1, 0.406 - 1] if input in range [-1, 1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229 * 2, 0.224 * 2, 0.225 * 2] if input in range [-1, 1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        #model = nn.DataParallel(model,device_ids=range(2))
        model.to(device)
        #model.eval()
        if isinstance(feature_layer, list):
            self.features = []
            self.featurelist = True
            i = 0
            previous_layer = 0
            for feature_ in feature_layer:
                a = nn.Sequential(*list(model.features.children())[previous_layer:(feature_ + 1)])
                a = nn.DataParallel(a, device_ids=range(2))
                a.to(device)
                self.features.append(a)
                previous_layer = feature_ + 1
                for k, v in self.features[i].named_parameters():
                    v.requires_grad = False
                i = i + 1
        else:
            self.featurelist = False
            self.features=(nn.Sequential(*list(model.features.children())[:(feature_layer + 1)]))
        # No need to BP to variable
            for k, v in self.features.named_parameters():
                v.requires_grad = False

    def forward(self, x):
        # Assume input range is [0, 1]
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        now_x = x
        if self.featurelist:
            output = []
            for i in range(5):
                result_layeri = self.features[i](now_x)
                output.append(result_layeri)
                now_x = result_layeri
        else:
            output = self.features(x)
        return output