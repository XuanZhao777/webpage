import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models import ResNet18_FSR
from attacks import FGSMAttack
from utils import get_pred

def evaluate_model():
    # Set device to CPU
    device = 'cpu'

    # Define transformation for the test dataset
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load the test dataset
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    # Load the pre-trained ResNet18_FSR model
    model = ResNet18_FSR(tau=0.1, num_classes=10, image_size=(32, 32))
    model.load_state_dict(torch.load('resnet18_fsr.pth', map_location=device))
    model.eval()

    # Set up the FGSM Attack
    fgsm_attack = FGSMAttack(model, epsilon=8/255)

    # Evaluation loop
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # FGSM Attack
            inputs = fgsm_attack.perturb(inputs, labels)

            # Model prediction
            outputs = model(inputs)
            _, predicted = get_pred(outputs, labels)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print('Accuracy on the test set: {:.2f}%'.format(accuracy))

    # Output the results for FGSM attack
    attack_accuracy = evaluate_fgsm_attack(model, testloader, fgsm_attack, device)
    print('Accuracy under FGSM attack: {:.2f}%'.format(attack_accuracy))

# Define the function to evaluate the model under FGSM attack
def evaluate_fgsm_attack(model, dataloader, fgsm_attack, device):
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # FGSM Attack
            inputs = fgsm_attack.perturb(inputs, labels)

            # Model prediction
            outputs = model(inputs)
            _, predicted = get_pred(outputs, labels)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# Run the evaluation
evaluate_model()
