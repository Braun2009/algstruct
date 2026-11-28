"""Run one small CPU forward pass. From the repository root: python -m examples.forward"""

import argparse

import torch

from model import ARMS, AlgebraConfig, AlgebraModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=ARMS, default='full')
    parser.add_argument('--compose-mode', choices=('rot', 'full_ap'), default='rot')
    args = parser.parse_args()

    torch.manual_seed(0)
    vocab = 32
    cfg = AlgebraConfig(
        width=64, layers=2, state_layers=(1,), k=4, q=16,
        arm=args.arm, compose_mode=args.compose_mode,
    )
    model = AlgebraModel(cfg, vocab=vocab).eval()
    events = torch.randint(vocab, (2, 8, 4))  # batch, time, input fields
    stats = {}
    with torch.inference_mode():
        logits = model(events, stats)

    print(f'arm={args.arm}  compose_mode={args.compose_mode}')
    print(f'events: {tuple(events.shape)}')
    print(f'logits: {tuple(logits.shape)}')
    if stats:
        shown = {k: round(v, 4) for k, v in stats.items()}
        print(f'structural stats: {shown}')
        print('  z_ratio is ||Z_T||_F / sqrt(k): 1.0 means the product is still an '
              'isometry, 0.0 means it has been flattened')
        print('  tau_mean is the mean proper-time step: 0.0 means every token was a no-op')


if __name__ == '__main__':
    main()
