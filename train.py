import torch


def train():
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
