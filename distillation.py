import torch
from torchvision import transforms, datasets
from torch import nn
import pytorch_lightning as lightning
from torch import optim
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    def __init__(self, epsilon, num_classes):
        super().__init__()
        self.eps = epsilon
        self.nc = num_classes
        self.inverse_eps = self.eps / (self.nc - 1)

    def forward(self, predictions, targets):
        smooth_targets = torch.zeros_like(predictions).scatter_(1, targets.view(-1, 1), 1.0)
        smooth_targets = smooth_targets * (1 - self.eps) + (1 - smooth_targets) * self.inverse_eps
        return -(smooth_targets * predictions).sum(dim=1).mean()


class JSDiv(nn.Module):
    def __init__(self, reduction):
        super().__init__()
        self.reduction = reduction

    def forward(self, p, q):
        pq = F.kl_div(F.log_softmax(p, dim=1), F.softmax(q, dim=1), reduction=self.reduction)
        qp = F.kl_div(F.log_softmax(q, dim=1), F.softmax(p, dim=1), reduction=self.reduction)
        return (pq + qp) * 0.5


class DistillModel(lightning.LightningModule):
    def __init__(self, teacher_model, student_model, temperature, alpha, batch_size,
                 learning_rate, workers, label_smoothing=False, epsilon=0.1, num_classes=100, use_jsdiv=False):
        super().__init__()

        self.teacher = teacher_model
        # Freeze the teacher model
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.student = student_model
        self.T = temperature
        self.alpha = alpha
        self.bs = batch_size
        self.lr = learning_rate
        self.workers = workers
        self.label_smoothing = label_smoothing
        self.epsilon = epsilon
        self.kldiv = nn.KLDivLoss(reduction='batchmean')
        self.jsdiv = JSDiv(reduction='batchmean')
        self.use_jsdiv = use_jsdiv
        if label_smoothing:
            self.nll = LabelSmoothingLoss(epsilon, num_classes)
        else:
            self.nll = nn.NLLLoss()

    def forward(self, x):
        teacher_out = self.teacher(x)
        student_out = self.student(x)
        return teacher_out, student_out

    def distillation_loss(self, teacher_out, student_out, y):
        if self.use_jsdiv:
            dark_knowledge = self.jsdiv(student_out / self.T, teacher_out / self.T)
        else:
            dark_knowledge = self.kldiv(
                F.log_softmax(student_out / self.T, dim=1),
                F.softmax(teacher_out / self.T, dim=1)
            )
        label_loss = self.nll(F.log_softmax(student_out, dim=1), y)
        total_loss = dark_knowledge * self.alpha * self.T * self.T + \
                     (1 - self.alpha) * label_loss
        return total_loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        teacher_out, student_out = self.forward(x)
        dloss = self.distillation_loss(teacher_out, student_out, y)
        return {'loss': dloss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        teacher_out, student_out = self.forward(x)
        dloss = self.distillation_loss(teacher_out, student_out, y)
        pred_idxs = torch.argmax(student_out.detach(), dim=1)
        return {'val_loss': dloss,
                'val_acc': (pred_idxs == y).float().mean()
                }

    def validation_end(self, outputs):
        avg_loss = torch.tensor([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.tensor([x['val_acc'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss, 'avg_val_acc': avg_acc}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.student.parameters(), lr=self.lr)
        steps = len(self.train_dataloader)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
        return [optimizer], [scheduler]

    @lightning.data_loader
    def train_dataloader(self):
        tfms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = datasets.ImageFolder('data/train/', transform=tfms)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.bs, shuffle=True,
                                                  num_workers=self.workers)
        return trainloader

    @lightning.data_loader
    def val_dataloader(self):
        tfms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        valset = datasets.ImageFolder('data/val/', transform=tfms)
        valloader = torch.utils.data.DataLoader(valset, batch_size=self.bs, shuffle=False,
                                                num_workers=self.workers)
        return valloader