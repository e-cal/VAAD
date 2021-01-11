import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms, models

DATA_DIR = "data/"


class AttentionClassifier():
    def __init__(self, cam=None):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load('attention_model.pth')
        self.model.eval()
        self.transforms = transforms.Compose([transforms.ToPILImage(),
                                              transforms.Resize(224),
                                              transforms.ToTensor(), ])

        if cam:
            self.cam = cam
            self.owns_cam = False
        else:
            self.owns_cam = True

    def init_cam(self):
        self.cam = cv2.VideoCapture(0)

    def overlay(self, frame, classification):
        cv2.putText(frame, classification, (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    def classify(self, frame, labels):
        img_tensor = self.transforms(frame).float()
        img_tensor = img_tensor.unsqueeze_(0)
        inputs = Variable(img_tensor).to(self.device)
        output = self.model(inputs)
        pred_ind = output.data.cpu().numpy().argmax()
        pred = labels[pred_ind]
        return pred

    def live(self):
        if not self.owns_cam:
            print("Cannot perform live classification with a pre-existing \
                  camera")
            return
        self.init_cam()
        while True:
            ret, frame = self.cam.read()
            pred = self.classify(frame, ['attentive', 'inattentive'])
            self.overlay(frame, pred)
            cv2.imshow('Attn Classifier', frame)
            if cv2.waitKey(1) == ord('q'):
                break
        self.cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    classifier = AttentionClassifier()
    classifier.live()
