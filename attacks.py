import torch
import torch.nn as nn

class FGSMAttack:
    def __init__(self, model, epsilon, _type='linf', num_iters=10, alpha=1.0):
        self.model = model
        self.epsilon = epsilon
        self._type = _type
        self.num_iters = num_iters
        self.alpha = alpha

    def perturb(self, original_images, labels):
        device = original_images.get_device()

        x = original_images.clone()
        x.requires_grad = True

        for _ in range(self.num_iters):
            with torch.enable_grad():
                outputs, _, _, _ = self.model(x, is_eval=True)
                cls_loss = nn.CrossEntropyLoss()(outputs, labels)

            grad_outputs = None
            grads = torch.autograd.grad(cls_loss, x, grad_outputs=grad_outputs, only_inputs=True)[0]

            x.data += self.alpha * torch.sign(grads.data)
            x = project(x, original_images, self.epsilon, self._type)
            x.clamp_(0, 1)  # Assuming the input data is normalized to [0, 1]

        return x.detach()

def project(x, original_x, epsilon, _type='linf'):
    if _type == 'linf':
        max_x = original_x + epsilon
        min_x = original_x - epsilon

        x = torch.max(torch.min(x, max_x), min_x)

    else:
        raise NotImplementedError

    return x
