from torch import nn
import torch.nn.functional as F

from pathlib import Path


class LinearModel():
    def __init__(self):
        self.name = "Linear_Model"

        abs_dir = Path(__file__).parent.absolute()
        models_dir = Path.joinpath(abs_dir, "models", self.name)
        models_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = models_dir

    class Model(nn.Module):
        def __init__(self, n_observations, n_actions):
            super(LinearModel, self).__init__()
            self.flatten = nn.Flatten()
            self.layer1 = nn.Linear(n_observations, 256)
            self.layer2 = nn.Linear(256, 512)
            self.layer3 = nn.Linear(512, 512)
            self.layer4 = nn.Linear(512, 256)
            self.layer5 = nn.Linear(256, n_actions)
            self.drop = nn.Dropout(0.2)

        def forward(self, x):
            x = self.flatten(x)
            x = F.relu(self.layer1(x))
            x = F.relu(self.layer2(x))
            x = self.drop(x)
            x = F.relu(self.layer3(x))
            x = self.drop(x)
            x = F.relu(self.layer4(x))
            return self.layer5(x)