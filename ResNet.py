import torch
import torchvision


class Model(torch.nn.Module):
  def __init__(self, n_outputs=4, freeze=False):
    super().__init__()
    # descargamos resnet
    resnet = torchvision.models.resnet18(weights='DEFAULT')
    # nos quedamos con todas las capas menos la última
    self.resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
    if freeze:
      for param in self.resnet.parameters():
        param.requires_grad=False
    # añadimos una nueva capa lineal para llevar a cabo la clasificación
    self.fc = torch.nn.Linear(512, n_outputs)
    self.dropoutj = torch.nn.Dropout(p=0.2)

  def forward(self, x):
    x = self.resnet(x)
    x = x.view(x.shape[0], -1)
    x = self.dropoutj(x)
    x = self.fc(x)
    return x

  def unfreeze(self):
    for param in self.resnet.parameters():
        param.requires_grad=True