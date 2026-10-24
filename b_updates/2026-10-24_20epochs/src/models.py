"""Member A baseline; outputs raw logits for cross-entropy loss."""

from torch import nn


class BaselineCNN(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        layers = []
        channels = 3
        for output_channels in (32, 64, 128):
            layers += [nn.Conv2d(channels, output_channels, 3, padding=1, bias=False),
                       nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
            channels = output_channels
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                        nn.Dropout(dropout), nn.Linear(128, num_classes))

    def forward(self, inputs):
        return self.classifier(self.features(inputs))


def build_model(num_classes=2, dropout=0.3):
    return BaselineCNN(num_classes=num_classes, dropout=dropout)
