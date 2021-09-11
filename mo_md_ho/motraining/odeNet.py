import torch

class odeNet(torch.nn.Module):
  def __init__(self, activation = None, input=2, layers=5, D_hid=128, output=1):
    super(odeNet,self).__init__()

    if activation is None:
      self.actF = torch.nn.Sigmoid()
    else:
      self.actF = activation

    self.fca = torch.nn.Sequential(
      torch.nn.Linear(D_hid, D_hid),
      self.actF
    )

    self.output = torch.nn.Sequential(
      torch.nn.Linear(D_hid, output)
    )

    self.input = torch.nn.Sequential(
      torch.nn.Linear(input, D_hid)
    )

    self.ffn = torch.nn.Sequential(
      self.input,
      *[self.fca for _ in range(layers)],
      self.output
    )
    self.layers = layers

    self.es = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(a+.1)) for a in range(output)])

  def forward(self,t):
    return self.ffn(t)

