import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, default='')
    parser.add_argument('--max_episodes', type=int, default=500)
    args = parser.parse_args()
    return args