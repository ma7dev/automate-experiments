import argparse
import json
import os
import pickle

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision.datasets import MNIST

from agi.model import AGINet


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=1000, metavar="N", help="input batch size for testing (default: 1000)"
    )
    parser.add_argument("--epochs", type=int, default=10, metavar="N", help="number of epochs to train (default: 10)")
    parser.add_argument("--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)")
    parser.add_argument("--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument("--save-model", action="store_true", default=False, help="For Saving the current Model")
    parser.add_argument("--load-model", action="store_true", default=False, help="For Loading the Model")
    parser.add_argument("--overfit", action="store_true", default=False, help="For Overfitting the Model")
    parser.add_argument("--load-history", action="store_true", default=False, help="For Loading History")
    args = parser.parse_args()
    return args


def train(args, model, device, train_loader, optimizer, epoch):
    history = {
        "loss": 0,
        "accuracy": 0,
    }
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        history["loss"] += loss.item()
        history["accuracy"] += (output.argmax(dim=1) == target).sum().item()
        if batch_idx % args.log_interval == 0:
            print(
                f"[Train] Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]"
                f"({100. * batch_idx / len(train_loader):.0f}%)\tLoss: {loss.item():.6f}"
            )
    history["loss"] = round(history["loss"] / len(train_loader.dataset), 4)
    history["accuracy"] = round((history["accuracy"] / len(train_loader.dataset)) * 100.0, 2)
    return history


def test(args, model, device, test_loader):
    history = {
        "loss": 0,
        "accuracy": 0,
    }
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            history["loss"] += test_loss
            history["accuracy"] += (output.argmax(dim=1) == target).sum().item()
    test_loss /= len(test_loader.dataset)
    print(
        f"[Test] Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}"
        f"({100.0 * correct / len(test_loader.dataset):.0f}%)\n"
    )
    history["loss"] = round(history["loss"] / len(test_loader.dataset), 4)
    history["accuracy"] = round((history["accuracy"] / len(test_loader.dataset)) * 100.0, 2)
    return history


def save_history_as_figure(history, output_path, prefix=""):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(history) + 1), [record["loss"] for record in history], label="Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"{output_path}/{prefix}_loss.png")
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(history) + 1), [record["accuracy"] for record in history], label="Accuracy")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.savefig(f"{output_path}/{prefix}_accuracy.png")


def save_best(history, output_path):
    index = history["test"].index(max(history["test"], key=lambda x: x["accuracy"]))
    best_record = {
        "train accuracy": history["train"][index]["accuracy"],
        "test accuracy": history["test"][index]["accuracy"],
        "train loss": history["train"][index]["loss"],
        "test loss": history["test"][index]["loss"],
    }
    with open(f"{output_path}/best_record.json", "w") as f:
        json.dump(best_record, f)


def main():
    output_path = "./logs"
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    args = read_args()
    history = None
    if not args.load_history:
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
        train_loader = torch.utils.data.DataLoader(
            MNIST(
                "../data",
                train=True,
                download=True,
                transform=torchvision.transforms.Compose(
                    [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))]
                ),
            ),
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs,
        )
        test_loader = torch.utils.data.DataLoader(
            MNIST(
                "../data",
                train=False,
                transform=torchvision.transforms.Compose(
                    [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))]
                ),
            ),
            batch_size=args.test_batch_size,
            shuffle=False,
            **kwargs,
        )
        model = AGINet().to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        if args.load_model:
            model.load_state_dict(torch.load("mnist_cnn.pth"))
        history = {"train": [], "test": []}
        for epoch in range(1, args.epochs + 1):
            train_history = train(args, model, device, train_loader, optimizer, epoch)
            test_history = test(args, model, device, test_loader)
            history["train"].append(train_history)
            history["test"].append(test_history)
        with open(f"{output_path}/history.pkl", "wb") as f:
            pickle.dump(history, f)
    else:
        with open(f"{output_path}/history.pkl", "rb") as f:
            history = pickle.load(f)
    print(history)
    save_history_as_figure(history["train"], output_path, "train")
    save_history_as_figure(history["test"], output_path, "test")
    save_best(history, output_path)
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pth")


if __name__ == "__main__":
    main()
