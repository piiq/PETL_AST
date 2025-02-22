#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 16:18:19 2023

@author: umbertocappellazzo
"""
import torch
from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, scheduler, device, criterion, epoch=None):
    model.train(True)

    loss = 0.0
    correct = 0
    total = 0

    desc = f"Training epoch {epoch}" if epoch is not None else "Training"
    progress_bar = tqdm(loader, desc=desc, leave=True)

    # Track batch-level metrics
    batch_metrics = []

    for idx_batch, (x, y) in enumerate(progress_bar):
        # Clear memory from previous batch
        if str(device) in ["cuda", "mps"]:
            (
                torch.cuda.empty_cache()
                if str(device) == "cuda"
                else torch.mps.empty_cache()
            )

        optimizer.zero_grad(set_to_none=True)  # More efficient than False

        x = x.to(device, non_blocking=True)  # Use non_blocking for potential speed-up
        y = y.to(device, non_blocking=True)

        outputs = model(x)
        loss_batch = criterion(outputs, y)

        # Calculate metrics before backward pass
        with torch.no_grad():
            batch_correct = (y == outputs.argmax(dim=-1)).sum().item()
            batch_acc = batch_correct / len(x)
            current_loss = loss_batch.item()

        loss += current_loss
        total += len(x)
        correct += batch_correct

        batch_metrics.append(
            {
                "batch": idx_batch,
                "loss": current_loss,
                "accuracy": batch_acc,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        # Backward pass and optimization
        loss_batch.backward()
        optimizer.step()
        scheduler.step()

        # Update progress bar
        progress_bar.set_postfix(
            {
                "loss": f"{current_loss:.4f}",
                "acc": f"{batch_acc*100:.2f}%",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            }
        )

        # Clear memory
        del x, y, outputs, loss_batch
        if str(device) in ["cuda", "mps"]:
            (
                torch.cuda.empty_cache()
                if str(device) == "cuda"
                else torch.mps.empty_cache()
            )

    loss /= len(loader)
    accuracy = correct / total

    return loss, accuracy, batch_metrics


def eval_one_epoch(model, loader, device, criterion, desc="Evaluating"):
    loss = 0.0
    correct = 0
    total = 0

    # Track batch-level metrics
    batch_metrics = []

    model.eval()

    progress_bar = tqdm(loader, desc=desc, leave=True)

    with torch.inference_mode():
        for idx_batch, (x, y) in enumerate(progress_bar):
            # Clear memory from previous batch
            if str(device) in ["cuda", "mps"]:
                (
                    torch.cuda.empty_cache()
                    if str(device) == "cuda"
                    else torch.mps.empty_cache()
                )

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            outputs = model(x)
            loss_batch = criterion(outputs, y)

            # Calculate metrics
            current_loss = loss_batch.item()
            batch_correct = (y == outputs.argmax(dim=-1)).sum().item()
            batch_acc = batch_correct / len(x)

            loss += current_loss
            total += len(x)
            correct += batch_correct

            batch_metrics.append(
                {"batch": idx_batch, "loss": current_loss, "accuracy": batch_acc}
            )

            # Update progress bar
            progress_bar.set_postfix(
                {"loss": f"{current_loss:.4f}", "acc": f"{batch_acc*100:.2f}%"}
            )

            # Clear memory
            del x, y, outputs, loss_batch
            if str(device) in ["cuda", "mps"]:
                (
                    torch.cuda.empty_cache()
                    if str(device) == "cuda"
                    else torch.mps.empty_cache()
                )

        loss /= len(loader)
        accuracy = correct / total

    return loss, accuracy, batch_metrics
