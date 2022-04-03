import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from skimage import transform
from matplotlib.pylab import plt

plt.rcParams['font.family'] = 'Arial Unicode MS'  # enable chinese character
pd.options.display.float_format = '{:,.8f}'.format


class Net(nn.Module):

    def __init__(self, class_number):
        super(Net, self).__init__()
        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # a. half-of-size

            # Defining another 2D convolution layer
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # a. half-of-size
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(64 * 5 * 5, class_number)  # 793 
            # nn.Linear(800, class_number)
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


class PredictModel:

    def __init__(self, model_path='model_best.pth', label_path='label.csv'):
        self.model_path = model_path
        # self.model = torch.load(self.model_path)
        self.model = Net(class_number=793)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.model.to('cpu')
        self.label2name_frame = pd.read_csv(label_path)
        self.id2name = dict(
            zip(self.label2name_frame.label, self.label2name_frame.name))
        self.image = None

    def process_image_np(self, image_path):
        image_raw = np.load(image_path)
        image_raw = image_raw[:, :, 0]
        image = transform.resize(image_raw, (40, 40))
        image = image.reshape(1, 1, image.shape[0], image.shape[1])
        image = torch.from_numpy(image).float()
        self.image = image  # update the image
        return image  # return image # torch.tensor

    def predict(self, image_path='test.npy'):
        image = self.process_image_np(image_path)
        pred = self.model(image)
        pred_label = pred.argmax(dim=1).item()
        pred_character = self.id2name[pred_label]

        return pred_label, pred_character

    def predict_top10(self, image_path='test.npy'):
        image = self.process_image_np(image_path)
        pred = self.model(image)
        prob_dist = torch.sigmoid(pred).detach().numpy()[0]
        prediction_top_10 = self.label2name_frame.copy()
        prediction_top_10['prob'] = prob_dist
        prediction_top_10 = prediction_top_10.sort_values(
            by='prob', ascending=False).head(10)
        return prediction_top_10

    def show_result(self, pred_label, pred_character):
        title = 'Predicted: {}\nTrue: {}'.format(pred_label, pred_character)
        plt.imshow(self.image.squeeze().numpy(), cmap='gray')
        plt.title(title)
        plt.show()


if __name__ == '__main__':
    # model_path = './models/model_mlp.pth'
    model_path = './models/model_best'
    label_path = './data/processed/label_name.csv'
    # Debug only
    image_path = './src/ui/test.npy'
    # load the model
    model = PredictModel(model_path, label_path)
    # make prediction
    pred_label, pred_character = model.predict(image_path)
    print("\nPredicted Label =", pred_label)
    print("\nChinese Character Label =", model.id2name[pred_label])
    # make top 10 prediction
    prediction_top_10 = model.predict_top10(image_path)
    print(prediction_top_10)
    # show the result
    model.show_result(pred_label, pred_character)
    print("Done Test")
