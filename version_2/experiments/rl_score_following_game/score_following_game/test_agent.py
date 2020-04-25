
import copy
import cv2
import os
import torch

import matplotlib.cm as cm
import numpy as np

from matplotlib.colors import LinearSegmentedColormap, Normalize
# from score_following_game.agents.human_agent import HumanAgent
# from score_following_game.agents.optimal_agent import OptimalAgent
from score_following_game.agents.networks_utils import get_network
from score_following_game.data_processing.data_pools import get_single_song_pool
from score_following_game.data_processing.utils import load_game_config
from score_following_game.environment.score_following_supervised_env import ScoreFollowingSupervisedEnv
from score_following_game.environment.render_utils import prepare_sheet_for_render, prepare_spec_for_render
from score_following_game.experiment_utils import initialize_trained_agent, get_make_env, make_env_tismir,\
    setup_evaluation_parser
from score_following_game.integrated_gradients import IntegratedGradients, prepare_grad_for_render
from score_following_game.reinforcement_learning.algorithms.models import Model
from score_following_game.utils import render_video, get_opencv_bar

from collections import OrderedDict

# render mode for the environment ('human', 'computer', 'video')
# render_mode = 'computer'
render_mode = 'video'
mux_audio = True

if __name__ == "__main__":
    """ main """

    parser = setup_evaluation_parser()
    parser.add_argument('--agent_type', help='which agent to test [rnn|lstm|gru|optimal].',
                        choices=['rnn', 'lstm', 'gru', 'optimal'], type=str, default="rnn")
    parser.add_argument('--use_cuda', help='if set use gpu instead of cpu.', action='store_true')
    
    args = parser.parse_args()

    # FIGURE OUT A WAY TO INCORPORATE THE OPTIMAL "AGENT" (answers)
    if args.agent_type != 'optimal':
        exp_name = os.path.basename(os.path.split(args.params)[0])

        if args.net is None:
            args.net = exp_name.split('-')[1]

        if args.game_config is None:
            args.game_config = 'game_configs/{}.yaml'.format(exp_name.split("-")[3].rsplit("_", 2)[0])

    config = load_game_config(args.game_config)

    pool = get_single_song_pool(
        dict(config=config, song_name=args.piece, directory=args.data_set, real_perf=args.real_perf, limit_song_steps = 500))

    observation_images = []

    # initialize environment -- MODIFY THIS
    env = make_env_tismir(pool, config, render_mode='video')

    # compile network architecture
    # n_actions = len(config["actions"])
    net = get_network("networks_{}".format(args.agent_type), args.net, 
        shapes=dict(perf_shape=config['spec_shape'], score_shape=config['sheet_shape']))

    # load network parameters
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")

    net.load_state_dict(torch.load(args.params, map_location=torch.device(device)))

    # set model to evaluation mode
    net.eval()

    # create agent
    use_cuda = torch.cuda.is_available()

    model = Model(net, optimizer=None)

    observation_images = []

    # get observations
    observation = env.reset()  # (perf, score)

    reward = 0
    done = False

    colors = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#000000', '#e0f3f8', '#abd9e9', '#74add1',
              '#4575b4', '#313695']
    colors = list(reversed(colors))
    cmap = LinearSegmentedColormap.from_list('cmap', colors)
    norm = Normalize(vmin=-1, vmax=1)

    pos_grads = []
    neg_grads = []
    abs_grads = []
    values = []
    tempo_curve = []

    while True:
        # feed state to model to get estimated pos
        model_in = OrderedDict()
        for obs_key in observation:
            model_in[obs_key] = torch.from_numpy(observation[obs_key]).float().unsqueeze(0).to(device)

        # import pdb; pdb.set_trace()
        # model_in["perf"].shape: torch.Size([1, 1, 78, 40])
        # model_in["score"].shape: torch.Size([1, 1, 80, 256]) => torch.Size([1, 1, 160, 512]) after changing score_factor to 1 from 0.5

        newPos = model(model_in)

        # perform step and observe
        observation, _, done, info = env.step(newPos)

        if env.obs_image is not None:
            bar_img = env.obs_image
            if render_mode == 'video':
                observation_images.append(bar_img)
            else:
                cv2.imshow("Stats Plot", bar_img)
                cv2.waitKey(1)

        if done:
            break

    # write video
    if render_mode == 'video':
        render_video(observation_images, pool, fps=config['spectrogram_params']['fps'], mux_audio=mux_audio,
                     real_perf=args.real_perf)
