import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from models import ResNet18_FSR
from attacks import FGSMAttack
from utils import adjust_learning_rate

# Set device to CPU
device = 'cpu'

# Define the dataset and data loader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

# Initialize the ResNet18_FSR model
model = ResNet18_FSR(tau=0.1, num_classes=10, image_size=(32, 32))
model.to(device)

# Set up the optimizer and criterion
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

# Set the number of epochs
num_epochs = 100

# Training loop
for epoch in range(num_epochs):
    model.train()
    adjust_learning_rate(optimizer, epoch)

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        # FGSM Attack
        fgsm_attack = FGSMAttack(model, epsilon=8/255)
        inputs = fgsm_attack.perturb(inputs, targets)

        # Forward pass
        outputs_tuple = model(inputs)
        outputs = outputs_tuple[0]  # Assuming the first element contains predictions
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print training information
        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Step [{batch_idx}/{len(trainloader)}], Loss: {loss.item()}')

# Save the trained model
torch.save(model.state_dict(), 'resnet18_fsr.pth')
