import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import numpy as np
from tqdm import tqdm

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data)

class DoubleConv(nn.Module):
    '''
    perform (conv => BatchNorm => ReLU) * 2
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True))
        
    def forward(self, x):
        x = self.conv(x)
        return x


class GlobalMaxPool2D(nn.Module):
    '''
    Reduce to feature space
    '''
    def __init__(self):
        super(GlobalMaxPool2D, self).__init__()
        
    def forward(self, x):
        x =  torch.max(torch.max(x, 3, keepdim=False)[0], 2, keepdim=False)[0]
        return x


class GlobalAvgPool2D(nn.Module):
    '''
    Reduce to feature space
    '''
    def __init__(self):
        super(GlobalAvgPool2D, self).__init__()
        
    def forward(self, x):
        x =  torch.mean(torch.mean(x, 3, keepdim=False), 2, keepdim=False)
        return x

class ResFeatureExtractor(nn.Module):
    def __init__(self, n_classes=14, intermediate_features=4096, 
                    pretrained_model=models.resnet50, 
                    use_half_precision=False, device='cpu'):
        super(ResFeatureExtractor, self).__init__()

        self.n_classes = n_classes
        self.intermediate_features = intermediate_features
        
        if False:#os.path.isfile(pretrained_model):
            checkpoint = torch.load(pretrained_model)
            pretrained_model = checkpoint['model']
        else:
            pretrained_model = pretrained_model(pretrained=True)
            pretrained_model = nn.Sequential(*list(pretrained_model.children())[0:8])
            for parameter in pretrained_model.parameters():
                parameter.requires_grad = True

            self.additional_stacked_layers = nn.Sequential(GlobalAvgPool2D(),
                                                            nn.ReLU(inplace=True),
                                                            nn.Linear(2048, 4096),
                                                            nn.Dropout2d(0.5),
                                                            nn.ReLU(inplace=True),
                                                            nn.Linear(4096, self.intermediate_features))

            self.pretrained = nn.Sequential(pretrained_model,
                                            self.additional_stacked_layers)
            
            self.classifier = nn.Sequential(nn.Dropout2d(0.5),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(self.intermediate_features, self.n_classes))
            
            if use_half_precision:
                self.make_half()


            self.to(device=device)
        


    def forward(self, x):
        # print('x = ', x.shape)
        x = self.pretrained(x)
        # print('x_1 = ', x.shape)
        # x = self.globalmaxpool(x)
        x = self.classifier(x)
        # print('x_2 = ', x.shape)
        # print('x_3 = ', x.shape)
        # print('x_4 = ', x.shape)

        # x = F.sigmoid(x)
        
        return x

    def get_parameters(self, with_classifier=True):
        if with_classifier:
            return list(self.pretrained.parameters()) + list(self.classifier.parameters())
        else:
            return list(self.pretrained.parameters())

    def get_parameters_additional(self, with_classifier=True):
        if with_classifier:
            return list(self.additional_stacked_layers.parameters()) + list(self.classifier.parameters())
        else:
            return list(self.additional_stacked_layers.parameters())

    def reset_weights(self, reset_fuct=weights_init):
        self.apply(reset_fuct)


    def make_half(self):
        self.half()
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()



class ResLSTM(nn.Module):
    def __init__(self, n_classes=14, pretrained_weights=None, lstm_size=512):
        super(ResLSTM, self).__init__()
        
        self.feature_extractor = ResFeatureExtractor()
        self.lstm_size = lstm_size
        self.n_classes = n_classes


        if pretrained_weights is not None:
            self.feature_extractor.load(pretrained_weights)
            self.feature_extractor = nn.Sequential(feature_extractor.modulelist[:-1])
            for parameter in self.feature_extractor.parameters():
                parameter.requires_grad = False



        # Using LSTM to classify phases
        self.lstm = nn.LSTM(self.feature_extractor.intermediate_features, self.lstm_size, batch_first=True)
        self.fc1 = nn.Linear(self.lstm_size, self.n_classes)

        


    def forward(self, x, hidden_state):
        NotImplementedError()
        # x = self.feature_extractor(x)
        # x = x.view(1, x.size(0), -1)
        # x, hidden_state = self.lstm(x, hidden_state)
        # x = x.view(x.size(1), -1)
        # x = self.fc1(x)

        # return x, hidden_state


    def get_parameters(self):
        return self.modulelist[1:].parameters()


    def predict(self, test_loader, score_type, max_iterations=30, viz=None):
        correct = 0
        total = 0
        first = True
        score = 0
        idx = 0
        self.eval()
        with torch.no_grad():
            for idx, (x, y) in enumerate(test_loader):
                # print(x.shape, y.shape)
                y_onehot = make_one_hot(y, 2)
                if torch.cuda.is_available():
                    x = x.cuda()
                    y_onehot = y_onehot.cuda()

                predicted = self.forward(x)
                predicted_softmax = F.softmax(predicted, dim=1)

            
                if score_type == 'accuracy':
                    score += self.calc_acc(predicted, y)
                elif score_type == 'dice':
                    score += self.calc_dice_score(predicted_softmax, y_onehot)

                if viz is not None:
                    image_args = {'normalize':True, 'range':(0, 1)}
                    # viz.show_image_grid(images=x.cpu()[:, 0, ].unsqueeze(1), name='Images_test')
                    viz.show_image_grid(images=y, name='TestGT')
                    viz.show_image_grid(images=predicted_softmax.cpu()[:, 0, ].unsqueeze(1), name='TestPred_1', image_args=image_args)
                    viz.show_image_grid(images=predicted_softmax.cpu()[:, 1, ].unsqueeze(1), name='TestPred_2', image_args=image_args)


                if idx==max_iterations:
                    break

            # print('Iteration: {}, Score: {}'.format(idx, score))

        return score/idx









def main():
    input_image = torch.empty(3, 3, 640, 640)
    unet = UNet(3, 2)
    output_image = unet(input_image)
    # print(output_image.shape)

    # nn.CrossEntropyLoss





