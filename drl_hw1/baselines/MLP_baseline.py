import numpy as np
import copy
import torch
import torch.nn as nn
from torch.autograd import Variable

class MLPBaseline:
  def __init__(self, env_spec, reg_coeff=1e-5):
    n = env_spec.observation_dim # number of states
    self._reg_coeff = reg_coeff

    self.model = nn.Sequential()
    self.model.add_module('mlp_0', nn.Linear(n+4, 32))
    self.model.add_module('tanh_0', nn.Tanh())
    self.model.add_module('mlp_1', nn.Linear(32, 1))
    
    self.learning_rate = 1e-4
    #self._coeffs = list(self.model.parameters())

  def _features(self, path):
    o = np.clip(path["observations"], -10, 10)
    if o.ndim > 2:
      o = o.reshape(o.shape[0], -1)
    l = len(path["rewards"])
    al = np.arange(l).reshape(-1, 1) / 1000.0
    feat = np.concatenate([o, al, al**2, al**3, np.ones((l, 1))], axis=1)
    return feat

  # https://github.com/jcjohnson/pytorch-examples/blob/master/nn/two_layer_net_nn.py
  def fit(self, paths, return_errors=False):

    featmat = np.concatenate([self._features(path) for path in paths])
    returns = np.concatenate([path["returns"] for path in paths])

    featmat = Variable(torch.from_numpy(featmat).float(), requires_grad=False)
    returns = Variable(torch.from_numpy(returns).float(), requires_grad=False)

    loss = torch.nn.MSELoss(size_average=True)
    for _ in range(100):
      predictions = self.model(featmat)
      errors = loss(predictions, returns)
      self.model.zero_grad()
      errors.backward()
      for param in self.model.parameters():
        param.data -= self.learning_rate * param.grad.data

    if return_errors:
      return 0, 0

  def predict(self, path):
    features = Variable(torch.from_numpy(self._features(path)).float(), requires_grad=False)
    pred = self.model(features).data.numpy()
    return np.squeeze(pred)

