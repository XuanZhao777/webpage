# attacks.py
import torch
import torch.nn as nn


class FGSMAttack:
    def __init__(self, model, epsilon):
        self.model = model
        self.epsilon = epsilon

    def perturb(self, x, y):
        # FGSM Attack
        x.requires_grad = True
        outputs = self.model(x)

        # Extract the actual tensor from the tuple
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        # Ensure the target tensor y is a 1D tensor
        if y.dim() > 1:
            y = y.squeeze(dim=1)

        loss = nn.CrossEntropyLoss()(outputs, y)
        gradient = torch.autograd.grad(loss, x)[0]
        x_adv = x + self.epsilon * gradient.sign()
        x_adv = torch.clamp(x_adv, 0, 1)
        return x_adv

