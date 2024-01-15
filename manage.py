import os
import sys
import os
import sys
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, input_features, output_features):
        super(Classifier, self).__init__()
        self.layer1 = nn.Linear(input_features, 512)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.layer2 = nn.Linear(512, 256)
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.layer3 = nn.Linear(256, 128)
        self.batchnorm3 = nn.BatchNorm1d(128)
        self.layer4 = nn.Linear(128, 64)
        self.batchnorm4 = nn.BatchNorm1d(64)
        self.layer5 = nn.Linear(64, output_features)

    def forward(self, x, training=True):
        x = self.layer1(x)
        x = self.batchnorm1(x)
        x = F.relu(x)
        x = F.dropout(x, training=training)
        x = self.layer2(x)
        x = self.batchnorm2(x)
        x = F.relu(x)
        x = self.layer3(x)
        x = self.batchnorm3(x)
        x = F.relu(x)
        x = self.layer4(x)
        x = self.batchnorm4(x)
        x = F.relu(x)
        x = self.layer5(x)
        x = F.softmax(x, dim=1)
        return x


def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'djangoProject.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
