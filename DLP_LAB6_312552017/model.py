import torch.nn as nn
from diffusers import UNet2DModel

class ClassConditionedUnet(nn.Module):
    def __init__(self, num_classes=24, class_emb_size=512):
        super(ClassConditionedUnet, self).__init__()

        self.sample_size = 64
        self.out_dim = 128
        # The underlying UNet model
        self.model = UNet2DModel(
            sample_size=self.sample_size,
            in_channels=3, 
            out_channels=3,
            block_out_channels=(self.out_dim, self.out_dim, self.out_dim*2, self.out_dim*2, self.out_dim*4, self.out_dim*4),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            class_embed_type=None
        )

        # The class embedding layer
        self.model.class_embedding = nn.Linear(num_classes, class_emb_size)

    def forward(self, sample, timestep, class_labels=None, return_dict=True):
        return self.model(sample, timestep, class_labels, return_dict)
