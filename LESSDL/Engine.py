# coding: utf-8

import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import time

def train_step(model: torch.nn.Module,
               data_loader: DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: str,
               epoch: int) -> float:
    model.train()
    running_loss = 0.0
    for batch, (reflectance, value) in enumerate(data_loader):
        reflectance = reflectance.to(device)
        model.train()
        output = model(reflectance)
        loss = loss_fn(output, value.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    loss = running_loss / len(data_loader)
    return loss


def train(model: torch.nn.Module,
          data_loader: DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: str):
    train_losses = []
    for epoch in tqdm(range(epochs)):
        start = time.time()
        train_loss = train_step(model, data_loader, loss_fn, optimizer, device, epoch)
        end = time.time()
        total_time = end - start
        print(f"\nEpoch {epoch + 1} of {epochs}, ETA: {total_time:.2f} Train Loss:{train_loss}")
        train_losses.append(train_loss)
    return train_losses
