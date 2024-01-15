import joblib
import pandas as pd
import pickle
import numpy as np
import random
import re
import nltk
import string
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import itertools
from django.shortcuts import render
from joblib import load
from tensorflow.python.keras.utils.np_utils import to_categorical

from TemperatureModel.Params_Temp import getingX
from TemperatureModel.Params_Temp import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn import preprocessing
from joblib import dump
from torch.utils.data import Dataset, DataLoader

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

value_list = [[2.00000000e+00, 3.19000000e+01, 2.16000000e+01, 5.22633972e+01,
               9.06047211e+01, 2.98506886e+01, 2.40350093e+01, 5.69188993e+00,
               5.19374478e+01, 2.25508198e-01, 2.51771373e-01, 1.59444059e-01,
               1.27727264e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
               0.00000000e+00, 3.76046000e+01, 1.27032000e+02, 4.47624000e+01,
               5.14100000e-01, 5.86931250e+03]]

param_ml_input = ['station', 'Present_Tmax', 'Present_Tmin', 'LDAPS_RHmin', 'LDAPS_RHmax', 'LDAPS_Tmax_lapse',
                  'LDAPS_Tmin_lapse', 'LDAPS_WS', 'LDAPS_LH', 'LDAPS_CC1', 'LDAPS_CC2', 'LDAPS_CC3', 'LDAPS_CC4',
                  'LDAPS_PPT1', 'LDAPS_PPT2', 'LDAPS_PPT3', 'LDAPS_PPT4', 'lat', 'lon', 'DEM', 'Slope',
                  'Solar radiation']
param_ml_output = ['Next_Tmax', 'Next_Tmin']


# Create your views here.
def Temperature_learning(request):
    data = pd.read_csv(r'E:\NoSql\djangoProject\SavedModels\Temperature dataset.csv')
    request.POST.get('learn')
    random_state = random.randint(1, 10000)

    X_data = data[param_ml_input]
    y_data = data[param_ml_output]
    X_scaled = getingX(X_data.values)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_data, test_size=0.2)
    y_train_min = np.array(y_train['Next_Tmin'])
    y_test_min = np.array(y_test['Next_Tmin'])
    y_train_max = np.array(y_train['Next_Tmax'])
    y_test_max = np.array(y_test['Next_Tmax'])
    X_train_min = np.array(X_train)
    X_test_min = np.array(X_test)
    X_train_max = X_train_min
    X_test_max = X_test_min

    max_iter = 100000

    # MIN model
    min_model = SGDRegressor(max_iter=max_iter, tol=0.01, random_state=random_state)
    min_model.fit(X_train_min, y_train_min)
    y_pred_min = min_model.predict(X_test_min)
    MIN_mse = mean_squared_error(y_test_min, y_pred_min)
    MIN_r2 = r2_score(y_test_min, y_pred_min)
    print(MIN_mse, MIN_r2)
    # MAX model
    max_model = SGDRegressor(max_iter=max_iter, tol=0.01, random_state=random_state)
    max_model.fit(X_train_max, y_train_max.ravel())
    y_pred_max = max_model.predict(X_test_max)
    MAX_mse = mean_squared_error(y_test_max, y_pred_max)
    MAX_r2 = r2_score(y_test_max, y_pred_max)
    print(MAX_mse, MAX_r2)

    return render(request, 'Temperature_learning.html',
                  {'MIN_MSE': round(MIN_mse, 2), 'MAX_MSE': round(MAX_mse, 2), 'MIN_R2': round(MIN_r2, 2),
                   'MAX_R2': round(MAX_r2, 2), 'Random_state': round(random_state, 2)})


def Temperature_predict(request):
    data = pd.read_csv(r'E:\NoSql\djangoProject\SavedModels\Temperature dataset.csv')
    # request.POST.get('RandX')
    cur_example = data.iloc[random.randint(1, len(data))]
    X_data = cur_example[param_ml_input]
    y_data = cur_example[param_ml_output]
    print(getingX([X_data.values]))
    print(y_data)
    model = load(r'E:\NoSql\djangoProject\SavedModels\Temperatur min model.joblib')
    y_pred_min = model.predict(getingX([X_data.values]))
    model = load(r'E:\NoSql\djangoProject\SavedModels\Temperatur max model.joblib')
    y_pred_max = model.predict(getingX([X_data.values]))
    print('Min:', y_pred_min)
    MIN_real = y_data['Next_Tmin']
    MAX_real = y_data['Next_Tmax']
    MIN_pred = y_pred_min
    MAX_pred = y_pred_max
    return render(request, 'Temperature_predict.html', {'MIN_real': MIN_real,
                                                        'MAX_real': MAX_real,
                                                        'MIN_pred': round(MIN_pred[0], 2),
                                                        'MAX_pred': round(MAX_pred[0], 2)})


def home(request):
    return render(request, 'HomePage.html')


def preprocess_text(text: str):
    x = (re.sub(r"[.,;$&!?=\\_`'-/:#~]+|[\d]+", " ",
                text.lower()))  # удаление пунктуации цифр и приведение к нижнему регистру
    x = x.translate({ord(i): None for i in '"{}%@^|+[]'})
    stop = nltk.corpus.stopwords.words('russian')
    cleartext = ' '.join([i for i in x.split() if i not in stop])
    sn = nltk.SnowballStemmer('russian')
    wn = nltk.WordNetLemmatizer()
    restext = [wn.lemmatize(sn.stem(i)) for i in cleartext.split()]  # лемматизация
    return restext


def vectorize(words, vocab):
    vector = torch.zeros(vocab.vocab_len, dtype=torch.float32)
    for l in words:
        vector[vocab.token_to_idx[l]] = 1
    return vector


class Vocab:
    def __init__(self, X):
        self.token = list(np.unique(list(itertools.chain.from_iterable([i for i in data['comment']]))))
        self.idx = range(len(self.token))
        self.idx_to_token = dict(zip(self.idx, self.token))
        self.token_to_idx = dict(zip(self.token, self.idx))
        self.vocab_len = len(self.idx_to_token)


class Classifier(nn.Module):
    def __init__(self, input_features, output_features):
        super(Classifier, self).__init__()
        self.layer1 = nn.Linear(input_features, 512)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.layer2 = nn.Linear(512, 256)
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.layer3 = nn.Linear(256, 128)
        self.batchnorm3 = nn.BatchNorm1d(128)
        self.layer4 = nn.Linear(128, 64)
        self.batchnorm4 = nn.BatchNorm1d(64)
        self.layer5 = nn.Linear(64, output_features)

    def forward(self, x, training=True):
        x = self.layer1(x)
        x = self.batchnorm1(x)
        x = F.relu(x)
        x = F.dropout(x, training=training)
        x = self.layer2(x)
        x = self.batchnorm2(x)
        x = F.relu(x)
        x = self.layer3(x)
        x = self.batchnorm3(x)
        x = F.relu(x)
        x = self.layer4(x)
        x = self.batchnorm4(x)
        x = F.relu(x)
        x = self.layer5(x)
        x = F.softmax(x, dim=1)
        return x


class ReviewDataset(Dataset):
    def __init__(self, X, y, vocab):
        self.X = X
        self.y = y
        self.vocab = vocab

    def vectorize(self, words):
        vector = torch.zeros(self.vocab.vocab_len, dtype=torch.float32)
        for l in words:
            vector[self.vocab.token_to_idx[l]] = 1
        return vector

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.vectorize(self.X[idx])
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return X, y


def test_accuracy(model, dataloader):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    accuracy = 0
    epoch_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            y_batch = torch.argmax(y_batch, axis=1)
            y_pred = torch.argmax(y_pred, axis=1)
            accuracy += torch.sum(y_batch == y_pred)
            epoch_loss += loss.item()
    accuracy = accuracy.item() / len(dataloader.dataset)
    return accuracy, epoch_loss


data = pd.read_csv(r'E:\NoSql\djangoProject\SavedModels\Toxic dataset.csv')
data_orig = pd.read_csv(r'E:\NoSql\djangoProject\SavedModels\Toxic origin.csv')
data['toxic'] = data['toxic'].astype(int)
data['comment'] = [str(i).split() for i in data['comment']]


def Toxic_learning(request):
    dataset = ReviewDataset([i for i in data['comment']], to_categorical(data.toxic, 2),
                            Vocab([i for i in data['comment']]))
    dataset_train_size = round(len(dataset) * 0.8)
    dataset_test_size = len(dataset) - dataset_train_size
    dataset_train, dataset_test = torch.utils.data.random_split(dataset, [dataset_train_size, dataset_test_size])
    batch_size = 2048
    num_epochs = 5
    num_classes = 2
    learning_rate = 0.01
    trainloader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(dataset=dataset_test, batch_size=batch_size)

    model = Classifier(dataset_train[0][0].size(0), dataset_train[0][1].size(0))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    total_step = len(trainloader)
    loss_list = []
    learning_process = 0
    train_epoch_res, test_epoch_res = '', ''
    if request.method == 'POST':
        button_pressed = request.POST.get('button_pressed')
        if button_pressed == "Start learning":
            print("обучение началось")
            for epoch in range(num_epochs):
                for i, (X, y) in enumerate(trainloader, 0):
                    outputs = model(X, training=True)
                    loss = criterion(outputs, y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total = y.size(0)
                train_accuracy_, train_loss = test_accuracy(model, trainloader)
                train_epoch_res = ('Epoch [{}/{}], Loss: {:.4f}, trainAccuracy: {:.2f}%'
                                   .format(epoch + 1, num_epochs, (train_loss / len(trainloader)),
                                           train_accuracy_ * 100))
                print(train_epoch_res)

                test_accuracy_, test_loss = test_accuracy(model, testloader)
                test_epoch_res = (
                    'Loss: {:.4f}, testAccuracy: {:.2f}%'.format((test_loss / len(testloader)), test_accuracy_ * 100))
                print(test_epoch_res)
                loss_list.append((test_loss / len(testloader)))

    return render(request, 'Toxic learning.html',
                  {'train_epoch_res': train_epoch_res, 'test_epoch_res': test_epoch_res})


def Toxic_predict(request):
    loaded_model = torch.load(r'E:\NoSql\djangoProject\SavedModels\Toxic_Teached_model.pt')
    sentense = ''
    toxic_result = ''
    if request.method == 'POST':
        button_pressed = request.POST.get('button_pressed')
        if button_pressed == 'Toxic predict':
            sentense = str(data_orig['comment'][random.randint(1, 10000)])
            print(sentense)
            try:
                pred = loaded_model(vectorize(preprocess_text(sentense), Vocab(data['comment'])).unsqueeze(0))
                if np.argmax(pred.detach().numpy(), axis=1):
                    toxic_result = 'Toxic'
                else:
                    toxic_result = 'Non toxic'
            except:
                toxic_result = 'В словаре отсутствуют набранные слова'
    return render(request, 'Toxic predict.html', {'sentense': sentense, 'toxic_result': toxic_result})
