import torch
import torch.nn as nn
import numpy as np


class BaseModel:

  def __init__(self, model, loss_fn, optimizer):

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model = model
    self.loss_fn = loss_fn
    self.optimizer = optimizer

    self.train_loader = None
    self.val_loader = None

    self.losses = []
    self.val_losses = []
    self.epochs = 0

  def set_loaders(self, train_loader, val_loader):
    self.train_loader = train_loader
    self.val_loader = val_loader

  def _training_step(self):

    def train(x, y):
      self.model.train()
      y_pred = self.model(x)
      loss = self.loss_fn(y_pred, y)
      loss.backward()
      self.optimizer.step()
      self.optimizer.zero_grad()
      return loss.item()
    return train

  def _validation_step(self):

    def validate(x, y):
      self.model.eval()
      y_pred = self.model(x)
      loss = self.loss_fn(y_pred, y)
      return loss.item()
    return validate

  def _mini_batch(self, validation=False):

    if validation:
      data_loader = self.val_loader
      step_fn = self._validation_step()
    else:
      data_loader = self.train_loader
      step_fn = self._training_step()

    mini_batch_losses = []
    for x_mini_batch, y_mini_batch in data_loader:
      x_mini_batch = x_mini_batch.to(self.device)
      y_mini_batch = y_mini_batch.to(self.device)

      loss = step_fn(x_mini_batch, y_mini_batch)
      mini_batch_losses.append(loss)
    return np.mean(mini_batch_losses)

  def train(self, epochs):
    for _ in range(epochs):

      loss = self._mini_batch()
      self.losses.append(loss)

      with torch.no_grad():
        val_loss = self._mini_batch(validation=True)
        self.val_losses.append(val_loss)


  def predict(self, X):
    self.model.eval()

    X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
    y_pred = self.model(X)

    self.model.train()

    return y_pred.detach().cpu().numpy()
