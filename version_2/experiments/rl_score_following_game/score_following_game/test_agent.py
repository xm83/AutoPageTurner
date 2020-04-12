
import copy
import cv2
import os
import torch

import matplotlib.cm as cm
import numpy as np

from matplotlib.colors import LinearSegmentedColormap, Normalize
from score_following_game.agents.human_agent import HumanAgent
from score_following_game.agents.optimal_agent import OptimalAgent
from score_following_game.agents.networks_utils import get_network
from score_following_game.data_processing.data_pools import get_single_song_pool
from score_following_game.data_processing.utils import load_game_config
from score_following_game.environment.score_following_env import ScoreFollowingEnv
from score_following_game.environment.render_utils import prepare_sheet_for_render, prepare_spec_for_render
from score_following_game.experiment_utils import initialize_trained_agent, get_make_env, make_env_tismir,\
    setup_evaluation_parser
from score_following_game.integrated_gradients import IntegratedGradients, prepare_grad_for_render
from score_following_game.reinforcement_learning.algorithms.models import Model
from score_following_game.utils import render_video, get_opencv_bar


# render mode for the environment ('human', 'computer', 'video')
# render_mode = 'computer'
render_mode = 'video'
mux_audio = True

if __name__ == "__main__":
    """ main """

    parser = setup_evaluation_parser()
    parser.add_argument('--agent_type', help='which agent to test [rl|optimal|human].',
                        choices=['rl', 'optimal', 'human'], type=str, default="rl")
    args = parser.parse_args()


    if args.agent_type == 'rl':
        exp_name = os.path.basename(os.path.split(args.params)[0])

        if args.net is None:
            args.net = exp_name.split('-')[1]

        if args.game_config is None:
            args.game_config = 'game_configs/{}.yaml'.format(exp_name.split("-")[3].rsplit("_", 2)[0])

    config = load_game_config(args.game_config)

    if args.agent_type == 'optimal':
        # the action space for the optimal agent needs to be continuous
        config['actions'] = []

    pool = get_single_song_pool(
        dict(config=config, song_name=args.piece, directory=args.data_set, real_perf=args.real_perf))

    observation_images = []

    # initialize environment
    env = make_env_tismir(pool, config, render_mode='human' if args.agent_type == 'human' else 'video')

    if args.agent_type == 'human' or args.agent_type == 'optimal':

        agent = HumanAgent(pool) if args.agent_type == 'human' else OptimalAgent(pool)
        alignment_errors, action_sequence, observation_images, episode_reward = agent.play_episode(env, render_mode)

    else:

        # compile network architecture
        n_actions = len(config["actions"])
        net = get_network("networks_sheet_spec", args.net, n_actions=n_actions,
                          shapes=dict(perf_shape=config['spec_shape'], score_shape=config['sheet_shape']))

        # load network parameters
        net.load_state_dict(torch.load(args.params))

        # set model to evaluation mode
        net.eval()

        # create agent
        use_cuda = torch.cuda.is_available()

        model = Model(net, optimizer=None)

        agent = initialize_trained_agent(model, use_cuda=use_cuda, deterministic=False)

        observation_images = []

        # get observations
        episode_reward = 0
        observation = env.reset()

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
            # choose action
            action = agent.select_action(observation)
            # perform step and observe
            observation, reward, done, info = env.step(action)
            episode_reward += reward

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
    if args.agent_type != 'human' and render_mode == 'video':
        render_video(observation_images, pool, fps=config['spectrogram_params']['fps'], mux_audio=mux_audio,
                     real_perf=args.real_perf)
