from torchvision.models import vgg19, resnet50, resnet18
import torch.nn as nn
import torch

class PretrainedWorkflowModel(nn.Module):
    def __init__(self, n_classes):
        self.model = self.build_model(n_classes)

    def build_model(self, n_classes, keep_layers=8):
        model = vgg19(pretrained=True)
        count = 8
        for i, params in model.named_children():
            if count<= keep_layers:
                params.requires_grad = False
            count += 1

        in_features = model.classifier[6].in_features
        
        classifiers = list(model.classifier.children())[:-1]
        classifiers.extend([nn.Linear(in_features=in_features,
                                        out_features=n_classes)])
        model.classifier = nn.Sequential(*classifiers)

        return model

    def forward(self, x):
        out = self.model.forward(x)

    def fit(self, train_loader, criterion, optimizer, epochs):
        iteration = 0
        for epoch in range(epochs):
            for iteration, (images, labels) in enumerate(train_loader):
                if torch.cuda.is_available():
                    images = images.cuda()
                    labels = labels.cuda()
                
                optimizer.zero_grad()
                outputs = self.forward(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                iteration += 1

                if iteration%500 == 0:
                    print('Epoch: {}, Iteration: {}, Loss: {}'.format(epoch, iteration, loss))
                    #accuracy = predict(test_loader)




