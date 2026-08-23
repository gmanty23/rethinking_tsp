"""
options.py

Central configuration and argument parsing for the project.

Base Architecture: Based on "Solving TSP Requires Rethinking Generalization" (Kool et al.)
 Contributions: Extensions for Asymmetric/Windy TSP including Asymmetric Node 
Embeddings (ANE), Neural Adaptive Bias (NAB), and directional GNN aggregation among others.
"""


from email import parser
import os
import time
import argparse
import torch


def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="Training and Evaluation configurations for NCO / Windy TSP"
    )

    # ---------------------------------------------------------
    # 1. Problem & Data Setup
    # ---------------------------------------------------------
    data_grp = parser.add_argument_group('Problem & Data Setup')
    data_grp.add_argument('--problem', default='tsp', 
                        help="Problem to solve: 'tsp', 'tspsl' (Supervised), or 'windy_tsp' (Asymmetric/Wind)")
    data_grp.add_argument('--min_size', type=int, default=20, help="Minimum size of the problem graph")
    data_grp.add_argument('--max_size', type=int, default=20, help="Maximum size of the problem graph")
    data_grp.add_argument('--neighbors', type=float, default=1, help="k-nearest neighbors for graph sparsification")
    data_grp.add_argument('--knn_strat', type=str, default='percentage', 
                        help="Graph Sparsification Strategy: 'percentage' / 'random_percentage' / 'cost_weighted_percentage'")
    data_grp.add_argument('--train_dataset', type=str, default=None, help="Dataset file for training (SL only)")
    data_grp.add_argument('--val_datasets', type=str, nargs='+', default=None, help="Dataset files for validation")
    data_grp.add_argument('--val_size', type=int, default=1000, help="Instances used for reporting validation performance")
    data_grp.add_argument('--data_distribution', type=str, default=None, help="Specific data distribution to generate")

    # ---------------------------------------------------------
    # 2. Base Model Architecture 
    # ---------------------------------------------------------
    model_grp = parser.add_argument_group('Base Model Architecture')
    model_grp.add_argument('--model', default='attention', help="Model type: 'attention' or 'nar'")
    model_grp.add_argument('--encoder', default='gnn', choices=['gat', 'gnn', 'mlp', 'aafm', 'edge_gat'],
                        help="Graph encoder architecture")
    model_grp.add_argument('--embedding_dim', type=int, default=128, help="Dimension of input embedding")
    model_grp.add_argument('--hidden_dim', type=int, default=128, help="Dimension of hidden layers in Enc/Dec")
    model_grp.add_argument('--n_encode_layers', type=int, default=3, help="Number of encoder/critic layers")
    model_grp.add_argument('--aggregation', default='max', help="Neighborhood aggregation: 'sum' / 'mean' / 'max'")
    model_grp.add_argument('--aggregation_graph', default='mean', help="Graph embedding aggregation: 'sum' / 'mean' / 'max'")
    model_grp.add_argument('--normalization', default='layer', help="Normalization type: 'batch' / 'layer' / None")
    model_grp.add_argument('--learn_norm', action='store_true', help="Enable learnable affine transformation during normalization")
    model_grp.add_argument('--track_norm', action='store_true', help="Enable tracking batch statistics during normalization")
    model_grp.add_argument('--gated', action='store_true', help="Enable edge gating during neighborhood aggregation")
    model_grp.add_argument('--n_heads', type=int, default=8, help="Number of attention heads")
    model_grp.add_argument('--tanh_clipping', type=float, default=10., help="Tanh clipping bounds for parameters. Set to 0 to disable.")
    model_grp.add_argument('--use_wind', action='store_true',
                        help="Inject explicit wind vectors (Wx, Wy) into the spatial coordinates.")
    model_grp.add_argument('--node_embedding_type', type=str, default='original', 
                            choices=['original', 
                                    'ane_pure', 
                                    'ane_hybrid', 
                                    'ane_no_gate', 
                                    'ane_3way_gate', 
                                    'ane_stats_only' 
                                    ],
                            help=("Choose node embedding strategy: "
                                "'original' (coords+stats)concatenated; "
                                "'ane_pure' (coords+local distances)gated; "
                                "'ane_hybrid' ((coords+local distances)gated+global stats)concatenated; "
                                "'ane_no_gate' (coords+local distances+global stats)concatenated); "
                                "'ane_3way_gate' (coords+local distances+global stats) gated; "
                                "'ane_stats_only' (coords+stats)gated"))
    model_grp.add_argument('--node_feature_type', type=str, default='coords', choices=['coords', 'learned', 'hybrid', 'blank', 'topo'],
                        help="Feature type for Windy TSP: 'coords' (x,y), 'learned' (stats), 'hybrid' (both), 'blank' (learned completely from blank parameter), or 'topo' (local neighborhood sampling only).")
    model_grp.add_argument('--gnn_direction_mode', type=str, default='forward', choices=['forward', 'backward', 'dual'],
                        help="Directional aggregation for GNN (vital for asymmetric edges): 'forward', 'backward', 'dual'.")
    model_grp.add_argument('--nab_mode', type=str, default='none', choices=['none', 'encoder', 'decoder', 'both', 'aafm'],
                        help="Neural Adaptive Bias (NAB) injection location.")
    model_grp.add_argument('--gnn_deep_bias', action='store_true', 
                        help="Inject NAB deeply into every GNN layer instead of just the initial layer.")

    # ---------------------------------------------------------
    # 4. Training & RL Hyperparameters
    # ---------------------------------------------------------
    train_grp = parser.add_argument_group('Training & RL Parameters')
    train_grp.add_argument('--n_epochs', type=int, default=100, help="Number of epochs to train")
    train_grp.add_argument('--epoch_size', type=int, default=1000000, help="Instances per epoch")
    train_grp.add_argument('--batch_size', type=int, default=128, help="Instances per batch")
    train_grp.add_argument('--accumulation_steps', type=int, default=1, help="Gradient accumulation steps (effective batch_size = batch_size * accumulation_steps)")
    train_grp.add_argument('--lr_model', type=float, default=1e-4, help="Learning rate for the actor network")
    train_grp.add_argument('--lr_critic', type=float, default=1e-4, help="Learning rate for the critic network")
    train_grp.add_argument('--lr_decay', type=float, default=1.0, help="Learning rate decay per epoch")
    train_grp.add_argument('--max_grad_norm', type=float, default=1.0, help="Maximum L2 norm for gradient clipping")
    train_grp.add_argument('--entropy_coeff', type=float, default=0.0, help="Coefficient for entropy regularization (to encourage exploration)")
    train_grp.add_argument('--exp_beta', type=float, default=0.8, help="Exponential moving average baseline decay")
    train_grp.add_argument('--baseline', default='rollout', help="Baseline to use: 'rollout', 'critic' or 'exponential'")
    train_grp.add_argument('--bl_alpha', type=float, default=0.05, help="Significance in t-test for updating rollout baseline")
    train_grp.add_argument('--bl_warmup_epochs', type=int, default=None, help="Number of warmup epochs for baseline")
    train_grp.add_argument('--rollout_size', type=int, default=10000, help="Instances used for updating rollout baseline")

    # ---------------------------------------------------------
    # 5. Environment & Logging
    # ---------------------------------------------------------
    env_grp = parser.add_argument_group('Environment & Logging')
    env_grp.add_argument('--eval_only', action='store_true', help="Run evaluation only (no training)")
    env_grp.add_argument('--seed', type=int, default=1234, help="Random seed")
    env_grp.add_argument('--num_workers', type=int, default=0, help="Number of DataLoader workers")
    env_grp.add_argument('--log_step', type=int, default=100, help="Log information every N steps")
    env_grp.add_argument('--log_dir', default='logs', help="TensorBoard output directory")
    env_grp.add_argument('--run_name', default='run', help="Name to identify the run")
    env_grp.add_argument('--output_dir', default='outputs', help="Directory to write output models to")
    env_grp.add_argument('--epoch_start', type=int, default=0, help="Starting epoch (relevant for LR decay)")
    env_grp.add_argument('--checkpoint_epochs', type=int, default=1, help="Save checkpoint every N epochs")
    env_grp.add_argument('--load_path', help="Path to load model parameters and optimizer state")
    env_grp.add_argument('--resume', help="Resume from previous checkpoint file")
    env_grp.add_argument('--checkpoint_encoder', action='store_true', help="Checkpoint encoder to decrease memory usage")
    env_grp.add_argument('--shrink_size', type=int, default=None, help="Shrink batch size if instances finish early")
    env_grp.add_argument('--no_tensorboard', action='store_true', help="Disable TensorBoard logging")
    env_grp.add_argument('--no_progress_bar', action='store_true', help="Disable progress bar")
    env_grp.add_argument('--no_cuda', action='store_true', help="Disable CUDA")


    opts = parser.parse_args(args)

    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S"))
    opts.save_dir = os.path.join(
        opts.output_dir,
        "{}_{}-{}".format(opts.problem, opts.min_size, opts.max_size),
        opts.run_name
    )
    if opts.bl_warmup_epochs is None:
        opts.bl_warmup_epochs = 1 if opts.baseline == 'rollout' else 0
    assert (opts.bl_warmup_epochs == 0) or (opts.baseline == 'rollout')
    
    return opts