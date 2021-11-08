import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser(description='CoGAIL')
    parser.add_argument(
        '--render-mode',
        default='headless',
        help='which visualization mode to use: headless or gui')
    parser.add_argument(
        '--algo',
        default='ppo',
        help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument(
        '--gail',
        action='store_true',
        default=True,
        help='do imitation learning with gail')
    parser.add_argument(
        '--gail-experts-dir',
        default='dataset/dataset-continuous-info-act',
        help='directory that contains expert demonstrations for gail')
    parser.add_argument(
        '--gail-batch-size',
        type=int,
        default=128,
        help='gail batch size')
    parser.add_argument(
        '--gail-epoch',
        type=int,
        default=5,
        help='gail epochs')
    parser.add_argument(
        '--recode_dim',
        type=int,
        default=102,
        help='input feature dim of the code reconstruction model')
    parser.add_argument(
        '--code_size',
        type=int,
        default=2,
        help='size of the code')
    parser.add_argument(
        '--lr',
        type=float,
        default=3e-4,
        help='learning rate')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer apha')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=True,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.0,
        help='entropy term coefficient')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients')
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='random seed')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--num-processes',
        type=int,
        default=1,
        help='how many training CPU processes to use')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=6000,
        help='number of forward steps')
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=10,
        help='number of ppo epochs')
    parser.add_argument(
        '--bc-pretrain-steps',
        type=int,
        default=30,
        help='number of bc pretrain steps')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=32,
        help='number of batches for ppo')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=1,
        help='log interval, one log per n updates')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=100,
        help='save interval, one save per n updates')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=30,
        help='eval interval, one eval per n updates')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=6000000,
        help='number of environment steps to train')
    parser.add_argument(
        '--env-name',
        default='cogail_exp1_2dfq',
        help='environment to train on')
    parser.add_argument(
        '--log-dir',
        default='/tmp/gym/',
        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument(
        '--save-dir',
        default='./trained_models/',
        help='directory to save agent logs')
    parser.add_argument(
        '--base-net-small',
        action='store_true',
        default=True,
        help='use smaller base net (works better on low dim controller)')
    parser.add_argument(
        '--use-cross-entropy',
        action='store_true',
        default=True,
        help='use cross entropy loss to update discriminator (works better on low dim controller)')
    parser.add_argument(
        '--use-curriculum',
        action='store_true',
        default=False,
        help='use curriculum step size')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=True,
        help='compute returns taking into account time limits')
    parser.add_argument(
        '--recurrent-policy',
        action='store_true',
        default=False,
        help='use a recurrent policy')
    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_true',
        default=True,
        help='use a linear schedule on the learning rate')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert args.algo in ['a2c', 'ppo', 'acktr']
    if args.recurrent_policy:
        assert args.algo in ['a2c', 'ppo'], \
            'Recurrent policy is not implemented for ACKTR'

    return args
