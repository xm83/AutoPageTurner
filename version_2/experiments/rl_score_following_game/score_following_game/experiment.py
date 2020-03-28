
import getpass
import os
import pickle
import torch

import numpy as np

from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from score_following_game.agents.networks_utils import get_network
from score_following_game.agents.optim_utils import get_optimizer, cast_optim_params
from score_following_game.data_processing.data_pools import get_data_pools, get_shared_cache_pools
from score_following_game.data_processing.data_production import create_song_producer, create_song_cache
from score_following_game.data_processing.utils import load_game_config
from score_following_game.evaluation.evaluation import PerformanceEvaluator as Evaluator
from score_following_game.experiment_utils import setup_parser, setup_logger, setup_agent, make_env_tismir, get_make_env
from score_following_game.reinforcement_learning.torch_extentions.optim.lr_scheduler import RefinementLRScheduler
from score_following_game.reinforcement_learning.algorithms.models import Model
from time import gmtime, strftime

if __name__ == '__main__':
    """ main """

    parser = setup_parser()
    args = parser.parse_args()

    np.random.seed(args.seed)

    # compile unique result folder
    time_stamp = strftime("%Y%m%d_%H%M%S", gmtime())
    tr_set = os.path.basename(args.train_set)
    config_name = os.path.basename(args.game_config).split(".yaml")[0]
    user = getpass.getuser()
    exp_dir = args.agent + "-" + args.net + "-" + tr_set + "-" + config_name + "_" + time_stamp + "-" + user

    args.experiment_directory = exp_dir

    # create model parameter directory
    args.dump_dir = os.path.join(args.param_root, exp_dir)
    if not os.path.exists(args.dump_dir):
        os.makedirs(args.dump_dir)

    args.log_dir = os.path.join(args.log_root, args.experiment_directory)

    # initialize tensorboard logger
    log_writer = None if args.no_log else setup_logger(args=args)

    args.log_writer = log_writer

    # cast optimizer parameters to float
    args.optim_params = cast_optim_params(args.optim_params)

    # load game config
    config = load_game_config(args.game_config)

    # initialize song cache, producer and data pools
    CACHE_SIZE = 50
    cache = create_song_cache(CACHE_SIZE)
    producer_process = create_song_producer(cache, config=config, directory=args.train_set, real_perf=args.real_perf)
    rl_pools = get_shared_cache_pools(cache, config, nr_pools=args.n_worker, directory=args.train_set)

    producer_process.start()

    env_fnc = make_env_tismir

    if args.agent == 'reinforce':
        env = get_make_env(rl_pools[0], config, env_fnc, render_mode=None)()
    else:
        env = ShmemVecEnv([get_make_env(rl_pools[i], config, env_fnc, render_mode=None) for i in range(args.n_worker)])

    # compile network architecture
    net = get_network('networks_sheet_spec', args.net, env.action_space.n,
                      shapes=dict(perf_shape=config['spec_shape'], score_shape=config['sheet_shape']))

    # load initial parameters
    if args.ini_params:
        net.load_state_dict(torch.load(args.ini_params))

    # initialize optimizer
    optimizer = get_optimizer(args.optim, net.parameters(), **args.optim_params)

    # initialize model
    model = Model(net, optimizer, max_grad_norm=args.max_grad_norm, value_coef=args.value_coef,
                  entropy_coef=args.entropy_coef)
    # use cuda if available
    if args.use_cuda:
        model.cuda()

    # load data from rl_pools
    dataset = []
    for pool in rl_pools:
        dataset += pool.get_data()
    train_ind = len(dataset) // 5 * 4
    train_data = dataset[:train_ind]
    test_data = dataset[train_ind:]
    cost_fxn = nn.MSELoss()
    
    num_epochs = 5
    for epoch in range(num_epochs):
        optimizer.zero_grad() # Clears existing gradients from previous epoch
        for score, audio, ans in train_data:
            score.to(device)
            audio.to(device)
            output = model(score, audio)
            loss = cost_fxn(output, ans)
            loss.backward() # Does backpropagation and calculates gradients
            optimizer.step() # Updates the weights accordingly
        if epoch%10 == 0:
            print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
            print("Loss: {:.4f}".format(loss.item()))

    # store the song history to a file
    if not args.no_log:
        with open(os.path.join(args.log_dir, 'song_history.pkl'), 'wb') as f:
            pickle.dump(producer_process.cache.get_history(), f)

    # stop the producer thread
    producer_process.terminate()

    if not args.no_log:
        log_writer.close()
