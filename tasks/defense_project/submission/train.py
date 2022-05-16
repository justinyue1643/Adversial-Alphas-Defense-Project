"""
The template for the students to train the model.
Please do not change the name of the functions in Adv_Training.
"""
import sys
sys.path.append("../../../")
import torch
import random
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from utils import get_dataset
import importlib.util
from CustomDataset import CustomDataset

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class VirtualModel:
    def __init__(self, device, model) -> None:
        self.device = device
        self.model = model

    def get_batch_output(self, images):
        predictions = []
        # for image in images:
        predictions = self.model(images).to(self.device)
            # predictions.append(prediction)
        # predictions = torch.tensor(predictions)
        return predictions

    def get_batch_input_gradient(self, original_images, labels):
        original_images.requires_grad = True
        self.model.eval()
        outputs = self.model(original_images)
        loss = F.nll_loss(outputs, labels)
        self.model.zero_grad()
        loss.backward()
        data_grad = original_images.grad.data
        return data_grad

class Adv_Training():
    """
    The class is used to set the defense related to adversarial training and adjust the loss function. Please design your own training methods and add some adversarial examples for training.
    The perturb function is used to generate the adversarial examples for training.
    """
    def __init__(self, device, file_path, target_label=None, epsilon=0.3, min_val=0, max_val=1):
        sys.path.append(file_path)
        from predict import LeNet
        self.model = LeNet().to(device)
        self.epsilon = epsilon
        self.device = device
        self.min_val = min_val
        self.max_val = max_val
        self.target_label = target_label
        self.nontarget_fgsm_perturb = self.load_perturb("../attacker_list/nontarget_FGSM")
        self.target_fgsm_perturb = self.load_perturb("../attacker_list/target_FGSM")
        self.pgd_perturb = self.load_perturb("../attacker_list/target_PGD")

    def load_perturb(self, attack_path):
        spec = importlib.util.spec_from_file_location('attack', attack_path + '/attack.py')
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)
        # for attack methods evaluator, the Attack class name should be fixed
        attacker = foo.Attack(VirtualModel(self.device, self.model), self.device, attack_path)
        return attacker


    def train(self, trainset, valset, device, epoches=40):
        self.model.to(device)
        self.model.train()
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=10)
        dataset_size = len(trainset)
        custom_dataset = CustomDataset(dataset_size)
        more_loader = torch.utils.data.DataLoader(custom_dataset, batch_size=100, shuffle=True, num_workers=10)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters())
        for epoch in range(epoches):  # loop over the dataset multiple times
            running_loss = 0.0
            # holder = [1,2,3,4,5]
            # t = [6, 7, 8]
            # (holder, *t)
            # print indices to figure out how long trainloader is
            # then conditional breakpoint to see how we can add back to the dataset

            for i, (inputs, labels) in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs = inputs.to(device)
                labels = labels.to(device)
                cpu_labels = labels.detach().cpu().tolist()
                l, freqs = np.unique(cpu_labels, return_counts=True)
                target_label = l[np.argmin(freqs)]

                # zero the parameter gradients
                nontarget_fgsm_adv_inputs, _ = self.nontarget_fgsm_perturb.attack(inputs, cpu_labels)
                nontarget_fgsm_adv_inputs = torch.tensor(nontarget_fgsm_adv_inputs).to(device)

                # target_fgsm_adv_inputs, _ = self.target_fgsm_perturb.attack(
                #     inputs, labels.detach().cpu().tolist(), target_label
                # )
                # target_fgsm_adv_inputs = torch.tensor(target_fgsm_adv_inputs).to(device)

                # tried setting target label to -1, broke cuda
                pgd_adv_inputs, _ = self.pgd_perturb.attack(inputs, labels.detach().cpu().tolist(), target_label)
                pgd_adv_inputs = torch.tensor(pgd_adv_inputs).to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                outputs = self.model(inputs)
                nontarget_fgsm_adv_outputs = self.model(nontarget_fgsm_adv_inputs)
                # target_fgsm_adv_outputs = self.model(target_fgsm_adv_inputs)
                pgd_adv_outputs = self.model(pgd_adv_inputs)

                # custom_dataset.append((adv_outputs, labels))

                loss = criterion(outputs, labels) + criterion(nontarget_fgsm_adv_outputs, labels) * 0.425 \
                       + criterion(pgd_adv_outputs, labels) * 0.575
                       # + criterion(target_fgsm_adv_outputs, labels) * 0.3 \

                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / dataset_size))
            running_loss = 0.0

        # more_loader = torch.utils.data.DataLoader(custom_dataset, batch_size=100, shuffle=True, num_workers=10)
        valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=True, num_workers=10)
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valloader:
                # print(inputs.shape, labels.shape)
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print("Accuracy of the network on the val images: %.3f %%" % (100 * correct / total))
        return


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adv_training = Adv_Training(device, file_path='.')
    dataset_configs = {
                "name": "CIFAR10",
                "binary": True,
                "dataset_path": "../datasets/CIFAR10/student/",
                "student_train_number": 10000,
                "student_val_number": 1000,
                "student_test_number": 100,
    }

    dataset = get_dataset(dataset_configs)
    trainset = dataset['train']
    valset = dataset['val']
    testset = dataset['test']
    adv_training.train(trainset, valset, device)
    torch.save(adv_training.model.state_dict(), "defense_project-model.pth")


if __name__ == "__main__":
    main()
