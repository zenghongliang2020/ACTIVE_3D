import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, default='ckpt_dir/')
    parser.add_argument('--max_episodes', type=int, default=1000)
    args = parser.parse_args()
    return args