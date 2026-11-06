import copy

import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import parametrize
from torch.utils.data import DataLoader, TensorDataset

from job_runner import cache_key, load_result, run_jobs, save_result

##### UTILITY FUNCTIONS #####
# The following functions are just utility functions to avoid code repetition. Their format is based on personal preference

def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train() 
    train_loss = 0.0
    total_samples = 0

    for batch, (X, y) in enumerate(dataloader):
        optimizer.zero_grad() # Must be set to zero each iteration to avoid exploding gradients
        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward() # Backpropagation to compute gradients
        optimizer.step() # Update the model parameters

    return

def evaluate(model, dataloader, loss_fn, device):
    model.eval() # Set the model to evaluation mode, this makes the model behave faster as no dropout, batchnorm, or gradient calculations are needed
    val_loss = 0.0
    total_samples = 0

    # The weights are fixed here, so the normalized (parametrized) weights are computed once instead of for every batch
    with torch.no_grad(), parametrize.cached(): # No gradient computation
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            val_loss += loss_fn(y_pred, y).item()*X.size(0) # Multiply by batch size to get the total loss for this batch
            total_samples += X.size(0)

    return val_loss / total_samples

def _train_key(model, train_loader, val_loader, optimizer, loss_fn, num_epochs, max_iter, device, train_eval_loader):
    # Hash of everything that goes into a training, or None if it cannot be determined (then it is not stored).
    # The random state is included, since the order of the shuffled batches depends on it.
    loaders = [(loader.dataset, loader.batch_size, type(loader.sampler).__name__, loader.generator)
               for loader in (train_loader, val_loader, train_eval_loader or train_loader)]
    try:
        return cache_key(_train, train_epoch, evaluate, model, loaders, optimizer.state_dict(), repr(optimizer),
                         repr(loss_fn), num_epochs, max_iter, str(device), torch.get_rng_state())
    except TypeError:
        return None

def train(model, train_loader, val_loader, optimizer, loss_fn, num_epochs, max_iter, device, train_eval_loader=None, verbose=True):
    # Trains the model, or loads the stored result if the same training was done before. A stored result also brings back
    # the optimizer and the random numbers as they were after the training, so later cells give the same results as well.
    key = _train_key(model, train_loader, val_loader, optimizer, loss_fn, num_epochs, max_iter, device, train_eval_loader)
    stored = load_result(key) if key is not None else None
    if stored is not None:
        model.load_state_dict(stored["model"])
        optimizer.load_state_dict(stored["optimizer"])
        torch.set_rng_state(stored["rng"])
        if train_loader.generator is not None:
            train_loader.generator.set_state(stored["generator"])
        if verbose:
            print(f"Loaded the stored training ({len(stored['train_losses'])} epochs). Final: Train Loss: {stored['train_losses'][-1]:.4f}, Val Loss: {stored['val_losses'][-1]:.4f}")
        return stored["train_losses"], stored["val_losses"]

    train_losses, val_losses = _train(model, train_loader, val_loader, optimizer, loss_fn, num_epochs, max_iter, device, train_eval_loader, verbose)

    if key is not None:
        save_result(key, {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": torch.get_rng_state(),
                          "generator": train_loader.generator.get_state() if train_loader.generator is not None else None,
                          "train_losses": train_losses, "val_losses": val_losses})
    return train_losses, val_losses

def _train(model, train_loader, val_loader, optimizer, loss_fn, num_epochs, max_iter, device, train_eval_loader=None, verbose=True):
    # train_eval_loader: optional loader over the training data used only for computing the training loss after each epoch.
    # Passing one with a large batch size makes this much cheaper than iterating the (small batch) train_loader again.
    if train_eval_loader is None:
        train_eval_loader = train_loader

    train_losses = []
    val_losses = []

    early_stopping_counter = 0
    best_val_loss = float('inf')
    best_state = None # Weights of the model at the epoch with the best validation loss

    for epoch in range(num_epochs):
        train_epoch(model, train_loader, optimizer, loss_fn, device)
        train_loss = evaluate(model, train_eval_loader, loss_fn, device)
        val_loss = evaluate(model, val_loader, loss_fn, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if verbose and (epoch % 50 == 0 or epoch == num_epochs - 1):
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= max_iter:
            if best_state is not None:
                model.load_state_dict(best_state) # Go back to the best weights instead of keeping the latest ones
            if verbose:
                print(f"Early stopping triggered. Restored the weights from epoch {epoch + 1 - early_stopping_counter}.")
            break

    return train_losses, val_losses

def predict(model, dataloader, device):
    model.eval()
    predictions = []

    with torch.no_grad(), parametrize.cached():
        for X, _ in dataloader:
            X = X.to(device)
            y_pred = model(X)
            predictions.append(y_pred.cpu().numpy())
    return np.concatenate(predictions, axis=0)


##### PARALLEL SWEEPS #####
# The networks here are small enough that one CPU core per model is as fast as using all of them, so independent
# trainings (e.g. a scan over Lipschitz constants, depths or widths) can run side by side in separate processes.

def fit(model_cls, model_kwargs, trn_data, val_data, test_data, lr, num_epochs, batch_size,
        max_iter=None, seed=42, eval_batch_size=4096, device="cpu"):
    # Trains a single model with Adam and MSE loss. The *_data arguments are (inputs, targets) tuples of tensors.
    # max_iter is the early stopping patience of train(), None means no early stopping.
    # Returns a dictionary with the per-epoch losses and the test predictions.
    if max_iter is None:
        max_iter = num_epochs
    torch.manual_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)

    trn_loader = DataLoader(TensorDataset(*trn_data), batch_size=batch_size, shuffle=True, generator=generator)
    trn_eval_loader = DataLoader(TensorDataset(*trn_data), batch_size=eval_batch_size, shuffle=False)
    val_loader = DataLoader(TensorDataset(*val_data), batch_size=eval_batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(*test_data), batch_size=eval_batch_size, shuffle=False)

    model = model_cls(**model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = _train(model, trn_loader, val_loader, optimizer, nn.MSELoss(), num_epochs, max_iter, device,
                                      train_eval_loader=trn_eval_loader, verbose=False)

    return {"train_losses": train_losses,
            "val_losses": val_losses,
            "predictions": predict(model, test_loader, device)}

def fit_one(job):
    return fit(**job)

def fit_many(jobs, n_workers=None):
    # Runs fit(**job) for every job in parallel and returns the results in the same order. At most job_runner.MAX_JOBS run
    # at a time, and results that are already stored are not trained again.
    # model_cls must be importable (i.e. defined in a module, not in the notebook) to be sent to the worker processes.
    return run_jobs(fit_one, jobs, n_workers=n_workers, label="Training", depends_on=(fit, _train, train_epoch, evaluate, predict))
