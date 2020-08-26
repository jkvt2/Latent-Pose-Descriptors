import torch
import torch.nn as nn

__all__ = ['vgg19']

class VGG19(nn.Module):

    def __init__(self, num_classes=1000):
        super(VGG19, self).__init__()
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.features0 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.features2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        self.features5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.features7 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        self.features10 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.features12 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.features14 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.features16 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        self.features19 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.features21 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.features23 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.features25 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        
        self.features28 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

    def forward(self, x):
        #we want 0, 5, 10, 19, 28 for vgg19
        feat0 = self.features0(x)
        x = self.relu(feat0)
        x = self.relu(self.features2(x))
        x = self.maxpool(x)
        feat5 = self.features5(x)
        x = self.relu(feat5)
        x = self.relu(self.features7(x))
        x = self.maxpool(x)
        feat10 = self.features10(x)
        x = self.relu(feat10)
        x = self.relu(self.features12(x))
        x = self.relu(self.features14(x))
        x = self.relu(self.features16(x))
        x = self.maxpool(x)
        feat19 = self.features19(x)
        x = self.relu(feat19)
        x = self.relu(self.features21(x))
        x = self.relu(self.features23(x))
        x = self.relu(self.features25(x))
        x = self.maxpool(x)
        feat28 = self.features28(x)
        return feat0, feat5, feat10, feat19, feat28

def vgg19(model_path, **kwargs):
    model = VGG19(**kwargs)
    state_dict = torch.load(model_path)
#        state_dict = load_state_dict_from_url(model_urls[arch],
#                                              progress=False)
    sdk = list(state_dict.keys())
    for k in sdk:
        if k.startswith('features'):
            if int(k.split('.')[1]) > 28:
                state_dict.pop(k)
            else:
                state_dict[k.replace('features.','features')] = state_dict.pop(k)
        else:
            state_dict.pop(k)
    model.load_state_dict(state_dict)
    return model