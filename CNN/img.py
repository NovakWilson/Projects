import struct
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


def read_idx_images(filename):
    with open(filename, 'rb') as f:
        m, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        data = np.fromfile(f, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols)
    return data


def read_idx_labels(filename):
    with open(filename, 'rb') as f:
        m, num_items = struct.unpack('>II', f.read(8))
        data = np.fromfile(f, dtype=np.uint8)
    return data


mnist_train_imgs = read_idx_images("MNIST/train-images.idx3-ubyte")
mnist_train_lbls = read_idx_labels("MNIST/train-labels.idx1-ubyte")
mnist_test_imgs = read_idx_images("MNIST/t10k-images.idx3-ubyte")
mnist_test_lbls = read_idx_labels("MNIST/t10k-labels.idx1-ubyte")

fashion_train_imgs = read_idx_images("FashionMNIST/train-images-idx3-ubyte")
fashion_train_lbls = read_idx_labels("FashionMNIST/train-labels-idx1-ubyte")
fashion_test_imgs = read_idx_images("FashionMNIST/t10k-images-idx3-ubyte")
fashion_test_lbls = read_idx_labels("FashionMNIST/t10k-labels-idx1-ubyte")


class MNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        img_tensor = torch.from_numpy(img).float().unsqueeze(0)
        img_tensor = img_tensor / 255.0

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, label


class ConvBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return x


class HeadMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class HeadFashion(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_epoch(loader, backbone, head, optimizer, criterion):
    backbone.train()
    head.train()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        optimizer.zero_grad()

        features = backbone(images)
        logits = head(features)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def eval_model(loader, backbone, head, criterion):
    backbone.eval()
    head.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            features = backbone(images)
            logits = head(features)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def first_plot(num_epochs, train_losses, test_losses, train_accs, test_accs):
    epochs_range = range(1, num_epochs + 1)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, test_losses, label='Test Loss')
    plt.title("Loss over epochs (для MNIST)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accs, label='Train Acc')
    plt.plot(epochs_range, test_accs, label='Test Acc')
    plt.title("Accuracy over epochs (для MNIST)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


def second_plot(num_epochs_fashion, train_losses_fashion, test_losses_fashion, train_accs_fashion, test_accs_fashion):
    epochs_range = range(1, num_epochs_fashion + 1)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses_fashion, label='Train Loss (Fashion)')
    plt.plot(epochs_range, test_losses_fashion, label='Test  Loss (Fashion)')
    plt.title("Loss over epochs (FashionMNIST Frozen backbone)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accs_fashion, label='Train Accuracy')
    plt.plot(epochs_range, test_accs_fashion, label='Test Accuracy')
    plt.title("Accuracy over epochs (FashionMNIST Frozen backbone)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


def third_plot(num_epochs_fashion_unfreeze,
               fash_train_losses, fash_test_losses,
               fash_train_accs, fash_test_accs,
               mnist_test_accs):

    epochs_range = range(1, num_epochs_fashion_unfreeze + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, fash_train_losses, label='Train Loss (Fashion)')
    plt.plot(epochs_range, fash_test_losses,  label='Test Loss (Fashion)')
    plt.title("FashionMNIST Loss (Unfreeze)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, fash_train_accs, label='Train Acc')
    plt.plot(epochs_range, fash_test_accs,  label='Test Acc')
    plt.title("FashionMNIST Accuracy (Unfreeze)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, mnist_test_accs, label='MNIST Test Acc')
    plt.title("MNIST Test Accuracy (Unfreeze)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim([0, 1])
    plt.legend()

    plt.tight_layout()
    plt.show()


def teach_mnist(backbone_, head_mnist_):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        list(backbone_.parameters()) + list(head_mnist_.parameters()),
        lr=1e-3
    )

    num_epochs = 5
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    for epoch in range(num_epochs):
        tr_loss, tr_acc = train_epoch(train_loader_mnist, backbone_, head_mnist_, optimizer, criterion)
        te_loss, te_acc = eval_model(test_loader_mnist, backbone_, head_mnist_, criterion)

        train_losses.append(tr_loss)
        train_accs.append(tr_acc)
        test_losses.append(te_loss)
        test_accs.append(te_acc)

        print(f"Epoch {epoch + 1}/{num_epochs}: "
              f"train_loss={tr_loss:.4f}, train_acc={tr_acc:.4f} | "
              f"test_loss={te_loss:.4f}, test_acc={te_acc:.4f}")

    first_plot(num_epochs, train_losses, test_losses, train_accs, test_accs)
    return backbone_, head_mnist_


def teach_fashion_frozen():
    checkpoint = torch.load("combined_model.pth")

    backbone = ConvBackbone()
    head_mnist = HeadMNIST()
    head_fashion = HeadFashion()

    backbone.load_state_dict(checkpoint["backbone"])
    head_mnist.load_state_dict(checkpoint["head_mnist"])
    head_fashion.load_state_dict(checkpoint["head_fashion"])

    for param in backbone.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer_fashion = optim.Adam(head_fashion.parameters(), lr=1e-3)

    num_epochs_fashion = 5
    train_losses_fashion = []
    train_accs_fashion = []
    test_losses_fashion = []
    test_accs_fashion = []

    for epoch in range(num_epochs_fashion):
        tr_loss, tr_acc = train_epoch(train_loader_fashion, backbone, head_fashion, optimizer_fashion, criterion)
        te_loss, te_acc = eval_model(test_loader_fashion, backbone, head_fashion, criterion)

        train_losses_fashion.append(tr_loss)
        train_accs_fashion.append(tr_acc)
        test_losses_fashion.append(te_loss)
        test_accs_fashion.append(te_acc)

        print(f"Epoch {epoch + 1}/{num_epochs_fashion} (Frozen backbone, training new head): "
              f"train_loss={tr_loss:.4f}, train_acc={tr_acc:.4f} | "
              f"test_loss={te_loss:.4f}, test_acc={te_acc:.4f}")

    second_plot(num_epochs_fashion, train_losses_fashion, test_losses_fashion, train_accs_fashion, test_accs_fashion)
    torch.save({
        'backbone': backbone.state_dict(),
        'head_mnist': head_mnist.state_dict(),
        'head_fashion': head_fashion.state_dict()
    }, "checkpoint_after_p6.pth")
    return backbone, head_mnist, head_fashion


def teach_fashion_unfreeze():
    checkpoint = torch.load("checkpoint_after_p6.pth")

    backbone = ConvBackbone()
    head_mnist = HeadMNIST()
    head_fashion = HeadFashion()

    backbone.load_state_dict(checkpoint["backbone"])
    head_mnist.load_state_dict(checkpoint["head_mnist"])
    head_fashion.load_state_dict(checkpoint["head_fashion"])

    for param in backbone.parameters():
        param.requires_grad = True

    optimizer_unfreeze = optim.Adam(
        list(backbone.parameters()) + list(head_fashion.parameters()),
        lr=1e-4
    )
    criterion_ = nn.CrossEntropyLoss()

    num_epochs_unfreeze = 5

    fash_train_losses = []
    fash_test_losses = []
    fash_train_accs = []
    fash_test_accs = []

    mnist_test_accs = []

    for epoch in range(num_epochs_unfreeze):
        tr_loss, tr_acc = train_epoch(train_loader_fashion, backbone, head_fashion, optimizer_unfreeze, criterion_)
        fash_train_losses.append(tr_loss)
        fash_train_accs.append(tr_acc)

        te_loss, te_acc = eval_model(test_loader_fashion, backbone, head_fashion, criterion_)
        fash_test_losses.append(te_loss)
        fash_test_accs.append(te_acc)

        _, mnist_acc = eval_model(test_loader_mnist, backbone, head_mnist, criterion_)
        mnist_test_accs.append(mnist_acc)

        print(f"Epoch {epoch+1}/{num_epochs_unfreeze} (Unfreezing backbone): "
              f"Fashion TrainAcc={tr_acc:.4f}, TestAcc={te_acc:.4f} | MNIST TestAcc={mnist_acc:.4f}")

    third_plot(num_epochs_unfreeze,
               fash_train_losses, fash_test_losses,
               fash_train_accs,  fash_test_accs,
               mnist_test_accs)

    torch.save({
        'backbone': backbone.state_dict(),
        'head_mnist': head_mnist.state_dict(),
        'head_fashion': head_fashion.state_dict()
    }, "checkpoint_after_p7.pth")

    return backbone, head_mnist, head_fashion


def teach_fashion_unfreeze_from_start():
    checkpoint = torch.load("combined_model.pth")
    backbone = ConvBackbone()
    head_mnist = HeadMNIST()
    head_fashion = HeadFashion()

    backbone.load_state_dict(checkpoint["backbone"])
    head_mnist.load_state_dict(checkpoint["head_mnist"])
    head_fashion.load_state_dict(checkpoint["head_fashion"])

    for param in backbone.parameters():
        param.requires_grad = True

    optimizer_unfreeze = optim.Adam(
        list(backbone.parameters()) + list(head_fashion.parameters()),
        lr=1e-3
    )

    criterion_ = nn.CrossEntropyLoss()

    num_epochs_unfreeze = 5

    fash_train_losses = []
    fash_test_losses = []
    fash_train_accs = []
    fash_test_accs = []
    mnist_test_accs = []

    for epoch in range(num_epochs_unfreeze):
        tr_loss, tr_acc = train_epoch(train_loader_fashion, backbone, head_fashion, optimizer_unfreeze, criterion_)
        fash_train_losses.append(tr_loss)
        fash_train_accs.append(tr_acc)

        te_loss, te_acc = eval_model(test_loader_fashion, backbone, head_fashion, criterion_)
        fash_test_losses.append(te_loss)
        fash_test_accs.append(te_acc)

        _, mnist_acc = eval_model(test_loader_mnist, backbone, head_mnist, criterion_)
        mnist_test_accs.append(mnist_acc)

        print(f"Epoch {epoch+1}/{num_epochs_unfreeze} (No freeze from start): "
              f"FashTrainAcc={tr_acc:.4f}, FashTestAcc={te_acc:.4f} | MNIST TestAcc={mnist_acc:.4f}")

    third_plot(num_epochs_unfreeze,
               fash_train_losses, fash_test_losses,
               fash_train_accs,  fash_test_accs,
               mnist_test_accs)

    torch.save({
        'backbone': backbone.state_dict(),
        'head_mnist': head_mnist.state_dict(),
        'head_fashion': head_fashion.state_dict()
    }, "checkpoint_after_p8.pth")

    return backbone, head_mnist, head_fashion


def find_confusions(backbone, head_fashion, test_fashion_dataset):
    backbone.eval()
    head_fashion.eval()

    best_match = {}
    for c in range(10):
        best_match[c] = {}
        for t in range(10):
            best_match[c][t] = (-1.0, None)

    with torch.no_grad():
        for idx in range(len(test_fashion_dataset)):
            image, label = test_fashion_dataset[idx]
            c = label

            image_batch = image.unsqueeze(0)
            features = backbone(image_batch)
            logits = head_fashion(features)
            probs = F.softmax(logits, dim=1)[0]

            for t in range(10):
                score_t = probs[t].item()
                old_score, old_idx = best_match[c][t]
                if score_t > old_score:
                    best_match[c][t] = (score_t, idx)

    return best_match


def show_confusions(best_match, test_fashion_dataset, nrows=10, ncols=10):
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 12))

    for c in range(nrows):
        for t in range(ncols):
            score, idx_img = best_match[c][t]
            img_tensor, label = test_fashion_dataset[idx_img]
            img_np = img_tensor.squeeze(0).cpu().numpy()

            ax = axes[c, t]
            ax.imshow(img_np, cmap='gray')
            ax.axis('off')
            ax.set_title(f"{c}->{t}\n{score:.2f}")

    plt.tight_layout()
    plt.show()


train_mnist_dataset = MNISTDataset(mnist_train_imgs, mnist_train_lbls)
test_mnist_dataset = MNISTDataset(mnist_test_imgs, mnist_test_lbls)
train_loader_mnist = DataLoader(train_mnist_dataset, batch_size=64, shuffle=True)
test_loader_mnist = DataLoader(test_mnist_dataset, batch_size=1000, shuffle=False)

train_fashion_dataset = MNISTDataset(fashion_train_imgs, fashion_train_lbls)
test_fashion_dataset = MNISTDataset(fashion_test_imgs,  fashion_test_lbls)
train_loader_fashion = DataLoader(train_fashion_dataset, batch_size=64, shuffle=True)
test_loader_fashion = DataLoader(test_fashion_dataset,  batch_size=1000, shuffle=False)


# backbone = ConvBackbone()
# head_mnist = HeadMNIST()
# criterion = nn.CrossEntropyLoss()
#backbone, head_mnist = teach_mnist(backbone, head_mnist)

#head_fashion = HeadFashion()
# torch.save({
#     'backbone': backbone.state_dict(),
#     'head_mnist': head_mnist.state_dict(),
#     'head_fashion': head_fashion.state_dict()
# }, "combined_model.pth")

#teach_fashion_frozen()
#teach_fashion_unfreeze()
#teach_fashion_unfreeze_from_start()


checkpoint = torch.load("checkpoint_after_p8.pth")
backbone = ConvBackbone()
head_mnist = HeadMNIST()
head_fashion = HeadFashion()

backbone.load_state_dict(checkpoint["backbone"])
head_mnist.load_state_dict(checkpoint["head_mnist"])
head_fashion.load_state_dict(checkpoint["head_fashion"])
best_match = find_confusions(backbone, head_fashion, test_fashion_dataset)
show_confusions(best_match, test_fashion_dataset)
# for c in range(10):
#     for t in range(10):
#         sc, idx_img = best_match[c][t]
#         print(f"Real={c}, Pred={t}, Score={sc:.4f}, idx={idx_img}")