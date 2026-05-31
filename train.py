import os
import time
from tqdm import tqdm
import torch
import torch.cuda.amp as amp
import math
import numpy as np
import pickle

from sklearn.utils.class_weight import compute_class_weight

from torch.utils.data import DataLoader, RandomSampler
from torch.nn import DataParallel

from utils.log_utils import log_values, log_values_sl
from utils.data_utils import BatchedRandomSampler
from utils import move_to




def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


def validate(model, dataset, problem, opts, baseline_cost=None):
    # Validate
    print('[{}] Validating...'.format(opts.run_name), flush=True)
    cost = rollout(model, dataset, opts)
    avg_cost = cost.mean()
    
    # 1. Try to use External Baseline (LKH) if provided
    if baseline_cost is not None:
        opt_gap = ((avg_cost / baseline_cost - 1) * 100)
        
        print('[{}] Val Avg Cost: {:.4f} +- {:.4f}'.format(
            opts.run_name, avg_cost, torch.std(cost)), flush=True)
        print('[{}] Val LKH Cost: {:.4f}'.format(
            opts.run_name, baseline_cost), flush=True)
        print('[{}] Val Gap: {:.3f}%'.format(
            opts.run_name, opt_gap), flush=True)
            
        return avg_cost, opt_gap

    # 2. Fallback: Try internal Ground Truth (for standard TSP)
    try:
        gt_cost = rollout_groundtruth(problem, dataset, opts)
        opt_gap = ((cost/gt_cost - 1) * 100)
        
        print('[{}] Val GT Cost: {:.4f} +- {:.4f}'.format(
            opts.run_name, gt_cost.mean(), torch.std(gt_cost)), flush=True)
        print('[{}] Val Avg Cost: {:.4f} +- {:.4f}'.format(
            opts.run_name, avg_cost, torch.std(cost)), flush=True)
        print('[{}] Val Gap: {:.3f}% +- {:.3f}'.format(
            opts.run_name, opt_gap.mean(), torch.std(opt_gap)), flush=True)
            
        return avg_cost, opt_gap.mean()

    except (KeyError, NotImplementedError, AttributeError):
        # 3. No Baseline found (Just report Cost)
        print('[{}] Val Avg Cost: {:.4f} +- {:.4f}'.format(
            opts.run_name, avg_cost, torch.std(cost)), flush=True)
        print('[{}] Val Gap: N/A (No LKH file or Ground Truth found)'.format(opts.run_name), flush=True)
        
        return avg_cost, 0


def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()
    
    def eval_model_bat(bat):
        with torch.no_grad():
            # Grab the cost matrix if it's available in the batch
            cost_mat = move_to(bat['cost_matrix'], opts.device) if 'cost_matrix' in bat else None
            
            # Pass cost_matrix to the model
            cost, _ = model(
                move_to(bat['nodes'], opts.device), 
                move_to(bat['graph'], opts.device),
                cost_matrix=cost_mat
            )
        return cost.data.cpu()

    return torch.cat([
        eval_model_bat(bat)
        for bat in tqdm(
            DataLoader(dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers), 
            disable=opts.no_progress_bar, ascii=True
        )
    ], 0)


def rollout_groundtruth(problem, dataset, opts):
    return torch.cat([
        problem.get_costs(bat['nodes'], bat['tour_nodes'])[0]
        for bat in DataLoader(
            dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers)
    ], 0)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_datasets, problem, tb_logger, opts):
    print("\n[{}] Start train epoch {}, lr={}".format(
    opts.run_name, epoch, optimizer.param_groups[0]['lr']), 
    flush=True
    )
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    if not opts.no_tensorboard:
        tb_logger.add_scalar('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    # Generate new training data for each epoch
    train_dataset = baseline.wrap_dataset(
        problem.make_dataset(
            min_size=opts.min_size, max_size=opts.max_size, batch_size=opts.batch_size, 
            num_samples=opts.epoch_size, distribution=opts.data_distribution, 
            neighbors=opts.neighbors, knn_strat=opts.knn_strat
        ))
    train_dataloader = DataLoader(
        train_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers)

    # Put model in train mode!
    model.train()
    optimizer.zero_grad()
    set_decode_type(model, "sampling")
    scaler = amp.GradScaler(enabled=True)

    # --- TRACKING ---
    epoch_confidence_sum = 0
    num_batches = 0
    # ----------------

    for batch_id, batch in enumerate(tqdm(train_dataloader, disable=opts.no_progress_bar, ascii=True)):
            batch_conf = train_batch(
                model,
                optimizer,
                baseline,
                epoch,
                batch_id,
                step,
                batch,
                tb_logger,
                opts,
                total_batches=len(train_dataloader),
                start_time=start_time,
                scaler=scaler       
            )

            epoch_confidence_sum += batch_conf
            num_batches += 1
            step += 1
    
    lr_scheduler.step(epoch)

    # --- PRINT EPOCH STATS ---
    avg_conf = epoch_confidence_sum / num_batches if num_batches > 0 else 0.0
    epoch_duration = time.time() - start_time
    print('[{}] Finished epoch {}, took {} s. Avg Confidence: {:.4f}'.format(
        opts.run_name, epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration)), avg_conf), 
        flush=True
    )
    # -------------------------

    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict()
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

    for val_idx, val_dataset in enumerate(val_datasets):
        
        # --- NEW CODE: Attempt to load LKH Baseline for this dataset ---
        baseline_cost = None
        try:
            # 1. Get the path of the current validation dataset
            # (Assumes opts.val_datasets is a list of paths matching the loaded datasets)
            val_path = opts.val_datasets[val_idx]
            dataset_basename = os.path.basename(val_path)
            
            # 2. Construct the expected path to the LKH result file
            # It looks in: results/lkh_windy/dataset_name.pkl
            lkh_file_path = os.path.join("results", "lkh_windy", dataset_basename)
            
            # 3. Load if exists
            if os.path.isfile(lkh_file_path):
                with open(lkh_file_path, 'rb') as f:
                    lkh_data = pickle.load(f)
                    # lkh_data is a list of tuples: (cost, tour, duration)
                    # We extract just the costs [row[0]] and take the mean
                    lkh_costs = [row[0] for row in lkh_data if row is not None]
                    baseline_cost = np.mean(lkh_costs)
        except Exception as e:
            print(f"Warning: Could not load LKH baseline: {e}")
        # ----------------------------------------------------------------

        # Pass the baseline_cost to the validate function
        avg_reward, avg_opt_gap = validate(model, val_dataset, problem, opts, baseline_cost=baseline_cost)
        
        if not opts.no_tensorboard:
            tb_logger.add_scalar('val{}/avg_reward'.format(val_idx+1), avg_reward, step)
            tb_logger.add_scalar('val{}/opt_gap'.format(val_idx+1), avg_opt_gap, step)

    baseline.epoch_callback(model, epoch)


def train_batch(model, optimizer, baseline, epoch, 
                batch_id, step, batch, tb_logger, opts, total_batches=None, start_time=None, scaler=None):
    # Unwrap baseline
    bat, bl_val = baseline.unwrap_batch(batch)
    
    # Optionally move Tensors to GPU
    x = move_to(bat['nodes'], opts.device)
    graph = move_to(bat['graph'], opts.device)
    cost_matrix = move_to(bat['cost_matrix'], opts.device) if 'cost_matrix' in bat else None
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # Evaluate model, get costs and log probabilities
    with amp.autocast(enabled=True):
            # Evaluate model, get costs and log probabilities
            if opts.entropy_coeff > 0:
                cost, log_likelihood, entropy = model(x, graph, cost_matrix=cost_matrix, return_entropy=True)
            else:
                cost, log_likelihood = model(x, graph, cost_matrix=cost_matrix, return_entropy=False)
                entropy = None

            # --- STATS: Calculate Model Confidence ---
            # log_likelihood is the sum of log_probs for the tour.
            # We approximate average probability per node.
            # (Note: This is an approximation for logging purposes)
            avg_log_prob = log_likelihood.mean() / x.size(1) 
            model_confidence = torch.exp(avg_log_prob).item()
            # -----------------------------------------

            # Evaluate baseline
            bl_val, bl_loss = baseline.eval(x, graph, cost) if bl_val is None else (bl_val, 0)

            # Calculate loss
            reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
            
            # Add Entropy Regularization
            if entropy is not None:
                # We want to MAXIMIZE entropy -> MINIMIZE -entropy
                # So we subtract (coeff * entropy) from the loss
                reinforce_loss = reinforce_loss - opts.entropy_coeff * entropy.mean()

            loss = reinforce_loss + bl_loss

    # Perform backward pass
    if scaler is not None:
            scaler.scale(loss).backward()
    else:
            loss.backward()
    
    # Clip gradient norms and get (clipped) gradient norms for logging
    if scaler is not None:
            scaler.unscale_(optimizer)

        # Clip gradient norms
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    
    # Perform optimization step after accumulating gradients
    if step % opts.accumulation_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad()

    # Logging
    if step % int(opts.log_step) == 0:
            log_values(cost, grad_norms, epoch, batch_id, step, log_likelihood, 
                    reinforce_loss, bl_loss, tb_logger, opts, 
                    total_batches=total_batches, start_time=start_time)
            # Print confidence to console for immediate check
            print(f"Conf: {model_confidence:.4f}", end=" ")
            
            if not opts.no_tensorboard:
                tb_logger.add_scalar('stats/confidence', model_confidence, step)
                if entropy is not None:
                    tb_logger.add_scalar('stats/entropy', entropy.mean().item(), step)

    return model_confidence
        
def train_epoch_sl(model, optimizer, lr_scheduler, epoch, train_dataset, val_datasets, problem, tb_logger, opts):
    print("\nStart train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    if not opts.no_tensorboard:
        tb_logger.add_scalar('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    # Create data loader with random sampling
    train_dataloader = DataLoader(train_dataset, batch_size=opts.batch_size, num_workers=opts.num_workers, 
                                  sampler=BatchedRandomSampler(train_dataset, opts.batch_size))

    # Put model in train mode!
    model.train()
    optimizer.zero_grad()
    set_decode_type(model, "greedy")

    for batch_id, batch in enumerate(tqdm(train_dataloader, disable=opts.no_progress_bar, ascii=True)):

        train_batch_sl(
            model,
            optimizer,
            epoch,
            batch_id,
            step,
            batch,
            tb_logger,
            opts
        )

        step += 1
    
    lr_scheduler.step(epoch)

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all()
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

    for val_idx, val_dataset in enumerate(val_datasets):
        avg_reward, avg_opt_gap = validate(model, val_dataset, problem, opts)
        if not opts.no_tensorboard:
            tb_logger.add_scalar('val{}/avg_reward'.format(val_idx+1), avg_reward, step)
            tb_logger.add_scalar('val{}/opt_gap'.format(val_idx+1), avg_opt_gap, step)
    

def train_batch_sl(model, optimizer, epoch, batch_id, 
                   step, batch, tb_logger, opts):
    # Optionally move Tensors to GPU
    x = move_to(batch['nodes'], opts.device)
    graph = move_to(batch['graph'], opts.device)
    
    if opts.model == 'nar':
        targets = move_to(batch['tour_edges'], opts.device)
        # Compute class weights for NAR decoder
        _targets = batch['tour_edges'].numpy().flatten()
        class_weights = compute_class_weight("balanced", classes=np.unique(_targets), y=_targets)
        class_weights = move_to(torch.FloatTensor(class_weights), opts.device)
    else:
        class_weights = None
        targets = move_to(batch['tour_nodes'], opts.device)
    
    # Evaluate model, get costs and loss
    cost, loss = model(x, graph, supervised=True, targets=targets, class_weights=class_weights)
    
    # Normalize loss for gradient accumulation
    loss = loss / opts.accumulation_steps
    
    # Perform backward pass
    loss.backward()
    
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    
    # Perform optimization step after accumulating gradients
    if step % opts.accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

    # Logging
    if step % int(opts.log_step) == 0:
        log_values_sl(cost, grad_norms, epoch, batch_id, 
                      step, loss, tb_logger, opts)
