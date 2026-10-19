"""ResNet18 with explicit freeze policy, including BatchNorm running statistics."""
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


class TransferResNet18(nn.Module):
    def __init__(self, mode='layer4', pretrained=False):
        super().__init__()
        if mode not in ('head', 'layer4', 'all'):
            raise ValueError(mode)
        self.mode = mode
        self.network = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.network.fc = nn.Linear(self.network.fc.in_features, 2)
        for name, parameter in self.network.named_parameters():
            parameter.requires_grad = (mode == 'all' or name.startswith('fc.') or
                                       (mode == 'layer4' and name.startswith('layer4.')))

    def train(self, mode=True):
        super().train(mode)
        if mode and self.mode != 'all':
            # src.engine calls model.train() each epoch. Keep frozen BN statistics fixed.
            self.network.eval()
            self.network.fc.train()
            if self.mode == 'layer4':
                self.network.layer4.train()
        return self

    def forward(self, inputs):
        return self.network(inputs)


def build(config, pretrained=False):
    if config['model'] == 'baseline_cnn':
        from src.models import build_model
        return build_model(2, config.get('dropout', 0.3))
    if config['model'] != 'resnet18':
        raise ValueError('Unsupported architecture')
    return TransferResNet18(config['mode'], pretrained)
