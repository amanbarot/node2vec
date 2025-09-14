import torch
import numpy as np
import time
from helper_classes import (
    word2vec, 
    cooccurenceDataset, 
    cooc_collate
    )
import os
import json
from datetime import datetime
# import importlib

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def get_logger(log_file_path, print_to_console=True):
    """
    Returns a logger that writes JSON-formatted logs to file (1 per line).
    
    Args:
        log_file_path (str): File to append logs to.
        print_to_console (bool): If True, also prints log entries to stdout.
    
    Returns:
        log_fn (callable): log_fn(message_dict: dict)
    """
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    def log_fn(message_dict):
        message_dict["timestamp"] = datetime.now().isoformat()
        line = json.dumps(message_dict)
        with open(log_file_path, "a") as f:
            f.write(line + "\n")
        if print_to_console:
            print(line)

    return log_fn

def save_checkpoint(model, optimizer, scheduler, epoch, batch_idx,
                    loss, save_dir, run_id,):

    run_dir = os.path.join(save_dir, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    ckpt_path = os.path.join(
        run_dir,
        f"ckpt_epoch_{epoch}_batch_{batch_idx}.pt"
    )

    checkpoint = {
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': (
            scheduler.state_dict() if scheduler is not None else None
        ),
        'loss': loss,
    }

    torch.save(checkpoint, ckpt_path)

def gen_embeddings(
    n, 
    g_walks, 
    t_L=1, 
    t_U=5, 
    embed_dim=64, 
    neg_sample_ct=5, 
    word_dist=None, 
    device=device, 
    batch_size=1000, 
    lr_range=[0.025, 0.0001], 
    n_epochs = 5,
    early_stopping_patience=np.inf,
    ):
    """
    """
    if word_dist is None:
        word_dist = np.ones(n)/n # change this
    w2v_inst = word2vec(n, embed_dim, word_dist, neg_sample_ct, device)
    w2v_inst.to(device)
    dat_set = cooccurenceDataset(g_walks, t_L, t_U)
    dat_ldr = torch.utils.data.DataLoader(
        dat_set, 
        batch_size, 
        shuffle=True, 
        collate_fn=cooc_collate
        )

    # Training initializations
    initial_lr, min_lr = lr_range[0], lr_range[1]
    if initial_lr <= min_lr:
        raise ValueError("Initial learning rate must be greater \
                         than minimum learning rate."
                         )
    optimizer = torch.optim.SGD(w2v_inst.parameters(), initial_lr)
    # start_time = time.time()
    neg_ll = np.zeros(n_epochs)
    last_batch_loss = 1000 # Arbitrary large value to start with
    cur_patience = early_stopping_patience

    # Setting up learning rate scheduler using LambdaLR
    n_train_steps = len(dat_set) * n_epochs // batch_size
    # cur_train_step = 0
    lr_fn = lambda step: max(
        (1 - step / n_train_steps) 
            * (1 - min_lr / initial_lr) 
            + min_lr / initial_lr,
        min_lr / initial_lr
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda=lr_fn
    )

    # Print and logging setup
    print_interval = 100
    cur_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = get_logger(f"logs/train_log_{cur_time}.json",
                        print_to_console=False)
    
    # Setup for saving checkpoints
    run_id = cur_time
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    save_ck_freq = n_train_steps // (n_epochs * 3)  # Save every 1/3 of an epoch

    # Training loop
    for epoch in range(n_epochs):
        for batch_idx, batch in enumerate(dat_ldr):
            optimizer.zero_grad()
            center, context = batch[0].to(device), batch[1].to(device)
            # center, context = batch[:, 0].to(device), batch[:, 1].to(device)
            batch_nll = w2v_inst(center, context)
            batch_loss = batch_nll.item()\
                /(center.shape[0] * (1 + neg_sample_ct))
            neg_ll[epoch] += batch_nll.item()
            batch_nll.backward()
            optimizer.step()
            lr_scheduler.step()  

            # Logging and early stopping
            if batch_idx % print_interval == 0:
                logger({
                    "epoch": epoch,
                    "batch": batch_idx,
                    "loss": batch_loss,
                    "lr": optimizer.param_groups[0]['lr'],
                })
                if last_batch_loss < batch_loss:
                    cur_patience -= 1
                    if cur_patience <= 0:
                        print("Early stopping at epoch %d, \
                              batch %d" % (epoch, batch_idx))
                        return {"co_occurences": dat_set,
                                "word2vec": w2v_inst,
                                "loss": neg_ll}
                else:
                    cur_patience = early_stopping_patience
                    last_batch_loss = batch_loss

            # Save checkpoint
            if batch_idx % save_ck_freq == 0 and batch_idx > 0:
                save_checkpoint(
                    model=w2v_inst,
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    loss=batch_loss,
                    save_dir=save_dir,
                    run_id=run_id
                )
        
        # End of epoch
        neg_ll[epoch] = neg_ll[epoch]/(len(dat_set) * (1 + neg_sample_ct))
        
        # Save checkpoint at end of epoch
        save_checkpoint(
            model=w2v_inst,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            epoch=epoch,
            batch_idx="epoch_end",
            loss=neg_ll[epoch],
            save_dir=save_dir,
            run_id=run_id
        )
    
    # End of training
    return({"co_occurences": dat_set,
            "word2vec": w2v_inst,
            "loss": neg_ll})

