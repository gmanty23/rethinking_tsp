# def log_values(cost, grad_norms, epoch, batch_id, step, log_likelihood, 
#                reinforce_loss, bl_loss, tb_logger, opts):
#     avg_cost = cost.mean().item()
#     grad_norms, grad_norms_clipped = grad_norms

#     # Log values to screen
#     print('\nepoch: {}, train_batch_id: {}, avg_cost: {}'.format(epoch, batch_id, avg_cost))

#     print('grad_norm: {}, clipped: {}'.format(grad_norms[0], grad_norms_clipped[0]))

#     # Log values to tensorboard
#     if not opts.no_tensorboard:
#         tb_logger.add_scalar('avg_cost', avg_cost, step)

#         tb_logger.add_scalar('actor_loss', reinforce_loss.item(), step)
#         tb_logger.add_scalar('nll', -log_likelihood.mean().item(), step)

#         tb_logger.add_scalar('grad_norm', grad_norms[0], step)
#         tb_logger.add_scalar('grad_norm_clipped', grad_norms_clipped[0], step)

#         if opts.baseline == 'critic':
#             tb_logger.add_scalar('critic_loss', bl_loss.item(), step)
#             tb_logger.add_scalar('critic_grad_norm', grad_norms[1], step)
#             tb_logger.add_scalar('critic_grad_norm_clipped', grad_norms_clipped[1], step)

import time

def log_values(cost, grad_norms, epoch, batch_id, step, log_likelihood, 
               reinforce_loss, bl_loss, tb_logger, opts, total_batches=None, start_time=None):
    avg_cost = cost.mean().item()
    grad_norms, grad_norms_clipped = grad_norms

    # Calculate Time/ETA
    time_stats = ""
    if start_time is not None and total_batches is not None and batch_id > 0:
        elapsed = time.time() - start_time
        progress = (batch_id + 1) / total_batches
        total_estimated = elapsed / progress
        eta = total_estimated - elapsed
        time_stats = " | ETA: {:.0f}m {:.0f}s".format(eta // 60, eta % 60)

    # --- CHANGE IS HERE ---
    # We strip the timestamp from run_name if it's too long, or just print it as is.
    # Assuming run_name is like "exp1_coords", we print it first.
    print('[{}] Ep {} | Bt {}/{} | Cost: {:.4f} | Grad: {:.2f}{}'.format(
        opts.run_name, epoch, batch_id, total_batches if total_batches else "?", 
        avg_cost, grad_norms[0], time_stats),
        flush=True
    )

    # Log values to tensorboard (unchanged)
    if not opts.no_tensorboard:
        tb_logger.add_scalar('avg_cost', avg_cost, step)
        tb_logger.add_scalar('actor_loss', reinforce_loss.item(), step)
        tb_logger.add_scalar('nll', -log_likelihood.mean().item(), step)
        tb_logger.add_scalar('grad_norm', grad_norms[0], step)
        tb_logger.add_scalar('grad_norm_clipped', grad_norms_clipped[0], step)

        if opts.baseline == 'critic':
            tb_logger.add_scalar('critic_loss', bl_loss.item(), step)
            tb_logger.add_scalar('critic_grad_norm', grad_norms[1], step)
            tb_logger.add_scalar('critic_grad_norm_clipped', grad_norms_clipped[1], step)

def log_values_sl(cost, grad_norms, epoch, batch_id, 
                  step, loss, tb_logger, opts):
    avg_cost = cost.mean().item()
    grad_norms, grad_norms_clipped = grad_norms

    # Log values to screen
    print('\nepoch: {}, train_batch_id: {}, loss: {}, avg_cost: {}'.format(epoch, batch_id, loss, avg_cost))

    print('grad_norm: {}, clipped: {}'.format(grad_norms[0], grad_norms_clipped[0]))
    
    # Log values to tensorboard
    if not opts.no_tensorboard:
        tb_logger.add_scalar('avg_cost', avg_cost, step)
        
        tb_logger.add_scalar('actor_loss', loss.item(), step)

        tb_logger.add_scalar('grad_norm', grad_norms[0], step)
        tb_logger.add_scalar('grad_norm_clipped', grad_norms_clipped[0], step)
