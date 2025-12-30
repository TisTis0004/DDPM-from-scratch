import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import DataLoader


def get_dataloader(batch_size=64, root="data/mnist", num_workers=4):
    """
    Downloads MNIST dataset into the "root" path if not already there, creates and returns a dataloader using that dataset
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=0.5, std=0.5)]
    )
    train_dataset = datasets.MNIST(
        root=root, train=True, download=True, transform=transform
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    return train_loader


if __name__ == "__main__":
    loader = get_dataloader()
    batch = next(iter(loader))
    print(f"Shape of the batch: {batch[0].shape}")  # [64, 1, 28, 28]
    print(
        f"Labels of the batch: {batch[1]}, shape: {batch[1].shape}"
    )  # [tensor of 64 labels][64]
    # All Good 🕺! for this step
