from torchvision.models import resnet18
import torch.nn as nn


def make_siamese():
    resnet_output = 1000
    distance_first_layer_size = 64
    distance_second_layer_size = 16
    output_size = 2

    image_encoder = resnet18(pretrained=True)
    distance_network = nn.Sequential(
        nn.Linear(resnet_output, distance_first_layer_size),
        nn.ReLU(inplace=True),
        nn.Linear(distance_first_layer_size, distance_second_layer_size),
        nn.Linear(distance_second_layer_size, output_size))

    return SiameseNetwork(image_encoder, distance_network)


class SiameseNetwork(nn.Module):
    def __init__(self, encoder, distance_network):
        super(SiameseNetwork, self).__init__()
        self.encoder = encoder
        self.distance_network = distance_network

    def forward_once(self, x):
        output = self.encoder(x)
        output = output.view(output.size()[0], -1)
        output = self.distance_network(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
