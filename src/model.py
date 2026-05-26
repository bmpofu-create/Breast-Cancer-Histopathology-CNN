import torch
import torch.nn as nn

class BaselineCNN(nn.Module):

    def __init__(self, dropout_rate=0.5):

        super(BaselineCNN,self).__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier=nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                256*8*8,
                512
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout_rate
            ),

            nn.Linear(
                512,
                1
            )
        )

    def forward(self,x):

        x=self.features(x)

        x=self.classifier(x)

        return x