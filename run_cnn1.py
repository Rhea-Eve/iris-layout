import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from iris_dataset import Iris  # Make sure this file exists and is correct

# ========= Model Definition =========
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(1040, 120)
        self.fc2 = nn.Linear(120, 84)
        self.size_fc = nn.Linear(2, 32)
        self.fc3 = nn.Linear(84 + 32, 3)

    def forward(self, x, size):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        size = size.view(-1, 2)
        size_features = F.relu(self.size_fc(size))
        if size_features.shape[0] != x.shape[0]:
            size_features = size_features.expand(x.shape[0], -1)
        x = torch.cat((x, size_features), dim=1)
        x = self.fc3(x)
        return x

# ========= Helper Functions =========
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def confidence_score(probabilities):
    top_prob = probabilities[0, 0].item()
    second_prob = probabilities[0, 1].item()
    return top_prob - second_prob

# ========= Main Script =========
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    net = Net()
    net.load_state_dict(torch.load('./iris_net.pth', map_location=device))
    net.to(device)
    net.eval()

    # Transforms
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load test dataset
    batch_size = 4
    testset = Iris(root='./data/imaging', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    classes = ('ff', 'logic', 'fill')

    # Run on one batch and display predictions
    dataiter = iter(testloader)
    images, labels, image_sizes = next(dataiter)
    images, image_sizes = images.to(device), image_sizes.to(device)

    with torch.no_grad():
        outputs = net(images, image_sizes)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

    print("\nBatch Results:")
    print("Ground Truth: ", ' '.join(f'{classes[labels[j]]}' for j in range(batch_size)))
    print("Predicted:    ", ' '.join(f'{classes[predicted[j]]}' for j in range(batch_size)))

    imshow(torchvision.utils.make_grid(images.cpu()))

    # Full test evaluation with accuracy and confidence
    correct = 0
    total = 0
    total_confidence = 0

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data in testloader:
            images, labels, image_sizes = data[0].to(device), data[1].to(device), data[2].to(device)
            outputs = net(images, image_sizes)

            probabilities = torch.softmax(outputs, dim=1)
            sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)

            for i in range(len(sorted_probs)):
                total_confidence += sorted_probs[i][0].item() - sorted_probs[i][1].item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    correct_pred[classes[label.item()]] += 1
                total_pred[classes[label.item()]] += 1

    avg_confidence = total_confidence / total
    accuracy_percent = 100 * correct / total

    print(f"\nAccuracy of the network on the test images: {accuracy_percent:.1f} % (Avg Confidence: {avg_confidence:.2f})")

    for classname in classes:
        if total_pred[classname] == 0:
            print(f"Accuracy for class: {classname:5s} is N/A (no samples)")
        else:
            class_acc = 100 * correct_pred[classname] / total_pred[classname]
            print(f"Accuracy for class: {classname:5s} is {class_acc:.1f} %")
