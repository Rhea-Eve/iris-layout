import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

import iris_dataset
#import cifar #NOTE: i dont think this is needed rn

# Current strategy:
#   Just try to tell between ff, logic, fill, other
#      - Reduce input channels from RGB to just gray - how to do that? This
#        should reduce the # of parameters we need to tune
#      - Refine the CNN to match our use case: right now the intermediate layers
#        are optimized for a task that's not ours (handwriting recognition)
#      - Maybe need to eliminate extremely small fill from the training set?
#      - Alternatively, do we specify a cell size? Need to think about what
#        that even means.
#          - Maybe what we want in the end is a classifier that
#            given a patch of image, guesses how many of what type of cell are in
#            a region with a certain probability?
#          - The underlying issue is that cell sizes are quite different in scale,
#            and the size of the cell matters. The problem is the current CNN
#            is designed explicitly to disregard scale (written numbers have
#            the same meaning regardless of size), so again, need to tune the CNN
#            to throw away the part that allows us to scale an object.

PATH = './iris_net.pth'

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) 
        #Input Channels (3): The model expects 3-channel input images (RGB).
        #Output Channels (6): Produces 6 feature maps by applying 6 filters.
        #Kernel Size (5): Each filter is 5x5 pixels.
        #Operation: Extracts low-level features such as edges and textures. 

        #Q what are we normalized to? what are the dimensions now?
        
        self.pool = nn.MaxPool2d(2, 2)
        #Pooling Type: Max Pooling.
        # Kernel Size (2x2): Each pooling operation considers a 2x2 region.
        # Stride (2): Moves the pooling window 2 pixels at a time.
        # Operation: Reduces the spatial dimensions of the image by half (downsampling), retaining only the most prominent features.


        self.conv2 = nn.Conv2d(6, 16, 5)
        #Input Channels (6): Takes the 6 feature maps produced by conv1.
        #Output Channels (16): Produces 16 feature maps by applying 16 filters.
        #Kernel Size (5): Each filter is 5x5 pixels.
        #Operation: Extracts higher-level features from the downsampled data.


        #self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc1 = nn.Linear(1040, 120) # BUT WHY

        #Input Features (1040): The flattened feature map from the convolutional and pooling layers.
        #Reason for 1040: The dimensions of the feature map depend on the input image size, convolutional layer settings, and pooling steps.
        #If the input image size is fixed, this value is manually calculated.
        #Output Features (120): Projects the input into a 120-dimensional space for further processing.
    
        self.fc2 = nn.Linear(120, 84)

       # Input Features (120): Takes the 120 features from fc1.
       # Output Features (84): Reduces dimensionality further.

        self.fc3 = nn.Linear(84, 3) # set to number of classes

        # Input Features (84): Takes the features from fc2.
        #Output Features (3): Outputs scores for 3 classes. Each score represents how likely the input belongs to a particular class.

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) 
        #Applies the first convolutional layer to the input tensor.
        #Applies the ReLU activation function element-wise.

        x = self.pool(F.relu(self.conv2(x)))
        #Applies the second convolutional layer to the tensor from the previous step.

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        #Converts the multi-dimensional tensor into a 1D tensor for fully connected layers.
        #Flattening starts from the second dimension (1), leaving the batch dimension (0) intact.
        #Input shape: [batch_size, 16, 5, 5].
        #Output shape: [batch_size, 16 * 5 * 5] (e.g., [4, 400]).

        x = F.relu(self.fc1(x))

        #Fully connected layer that transforms the flattened features into a 120-dimensional vector.
        #Input shape: [batch_size, 400].
        #Output shape: [batch_size, 120].
        #Applies ReLU activation for non-linearity.

        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #Maps the 84-dimensional vector to 3 output scores, one for each class.
        #Input shape: [batch_size, 84].
        #Output shape: [batch_size, 3].
        return x

if __name__ == "__main__":

#Transforms: Prepares the input images for training by converting them to tensors and normalizing them.
#Batch Size: Sets the number of samples per batch for training and testing to 4.

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    #debugset = cifar.CIFAR10(root='./data', train=True, download=True, transform=transform)
    #debugloader = torch.utils.data.DataLoader(debugset, batch_size=batch_size, shuffle=True, num_workers=2)
    import pickle
    from typing import Any
    data: Any = []
    targets = []

    trainset = iris_dataset.Iris(root='./imaging', train=True,
                                            download=True, transform=transform)
    print(len(trainset.classes))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = iris_dataset.Iris(root='./imaging', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    
    #Loads the training and testing datasets using the custom iris_dataset class.

    classes = ('ff', 'logic', 'fill')

    #DataLoader: Wraps the datasets for easy iteration in batches.
    #Defines class labels.


    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # print images
    print('Image check: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
    #imshow(torchvision.utils.make_grid(images)) #commented out to run faster for testing

    if True:
        net = Net()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # Assuming that we are on a CUDA machine, this should print a CUDA device:
        print(f"this is the device {device}")
        net.to(device)

        criterion = nn.CrossEntropyLoss() #this is the loss function!!!! LogSoftmax and Negative Log-Likelihood Loss
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        #Training Loop

        for epoch in range(4):  # loop over the dataset multiple times
            print(f"Entering epoch {epoch}")

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                # inputs, labels = data
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0


        #Logs the average loss every 2000 mini-batches.


        print('Finished Training')

        torch.save(net.state_dict(), PATH)
    if True:
        net = Net()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # Assuming that we are on a CUDA machine, this should print a CUDA device:
        print(device)
        net.load_state_dict(torch.load(PATH, weights_only=True))
        net.to(device)

        dataiter = iter(testloader)
        images, labels = next(dataiter)

        # print images
        print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
        #imshow(torchvision.utils.make_grid(images)) #commented out to run faster for testing

        images_cuda = images.to(device)
        outputs = net(images_cuda)
        _, predicted = torch.max(outputs, 1)

        print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                                    for j in range(4)))

        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                # calculate outputs by running images through the network
                images_cuda = images.to(device)
                labels_cuda = labels.to(device)
                outputs = net(images_cuda)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels_cuda.size(0)
                correct += (predicted == labels_cuda).sum().item()

        print(f'Accuracy of the network on the test images: {100 * correct // total} %')

        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        # again no gradients needed
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images_cuda = images.to(device)
                labels_cuda = labels.to(device)
                outputs = net(images_cuda)
                _, predictions = torch.max(outputs, 1)
                # collect the correct predictions for each class
                for label, prediction in zip(labels_cuda, predictions):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1


        # print accuracy for each class
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')