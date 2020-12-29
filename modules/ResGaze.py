import torch.nn as nn

from modules.renset import resnet50

class ResGaze(nn.Module):
    def __init__(self):
        super(ResGaze, self).__init__()
        self.gaze_network = resnet50(pretrained=True)
        self.gaze_fc = nn.Sequential(
            nn.Linear(2048, 2),
        )

    def forward(self, x):
        feature = self.gaze_network(x)
        feature = feature.view(feature.size(0), -1)
        gaze = self.gaze_fc(feature)

        return gaze