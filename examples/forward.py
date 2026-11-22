"""Run one small CPU forward pass. From the repository root: python -m examples.forward"""

import argparse

import torch

from model import ARMS, AlgebraConfig, AlgebraModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=ARMS, default='full')
    args = parser.parse_args()

    torch.manual_seed(0)
    vocab = 32
    cfg = AlgebraConfig(width=64, layers=2, state_layers=(1,), k=3, arm=args.arm)
    model = AlgebraModel(cfg, vocab=vocab).eval()
    events = torch.randint(vocab, (2, 8, 4))  # batch, time, input fields
    with torch.inference_mode():
        logits = model(events)

    print(f'arm={args.arm}')
    print(f'events: {tuple(events.shape)}')
    print(f'logits: {tuple(logits.shape)}')


if __name__ == '__main__':
    main()
