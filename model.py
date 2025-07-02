import subprocess
import sys
import pathlib
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--extra-index-url", "https://download.pytorch.org/whl/cpu"])

try:
    import torch
except ImportError:
    install("typing-extensions==4.7.1")
    install("torch==1.13.0+cpu")


import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
import numpy as np
#from sklearn.metrics import roc_auc_score
#from sklearn.model_selection import train_test_split
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.utils import gen_batches
import torch.nn as nn
import torch
import torch.nn.functional as F

filepath = pathlib.Path(__file__).parent.resolve()
#class RandomForestQuantifier(_forest.BaseForest):
#    def __init__(self, n_estimators=100, max_depth=10, min_samples_leaf=100,
#                        criterion="squared_error",
#                        min_samples_split=2,
#                        min_weight_fraction_leaf=0.0,
#                        max_features=1.0,
#                        max_leaf_nodes=None,
#                        min_impurity_decrease=0.0,
#                        bootstrap=False,
#                        oob_score=False,
#                        n_jobs=None,
#                        random_state=None,
#                        verbose=0,
#                        warm_start=False,
#                        ccp_alpha=0.0,
#                        max_samples=None,
#                        monotonic_cst=None):
#        self.max_depth = max_depth
#        self.min_samples_leaf = min_samples_leaf
#        estimator = QuantifierTree(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
#        super().__init__(n_estimators=n_estimators, estimator=estimator)
#        self.criterion = criterion
#        self.max_depth = max_depth
#        self.min_samples_split = min_samples_split
#        self.min_samples_leaf = min_samples_leaf
#        self.min_weight_fraction_leaf = min_weight_fraction_leaf
#        self.max_features = max_features
#        self.max_leaf_nodes = max_leaf_nodes
#        self.min_impurity_decrease = min_impurity_decrease
#        self.ccp_alpha = ccp_alpha
#        self.monotonic_cst = monotonic_cst
#
#                        
#
#    def _set_oob_score_and_attributes(self, X, y):
#        # Fit the base estimator
#        self.estimator_.fit(X, y)
#        # Get the out-of-bag score
#        self.oob_score_ = self.estimator_.oob_score_
#        # Get the feature importances
#        self.feature_importances_ = self.estimator_.feature_importances_
#
#    def predict(self, X):
#        # Predict using the base estimator
#        preds = []
#        for i in range(self.n_estimators):
#            preds.append(self.estimators_[i].predict(X))
#        return np.mean(preds, axis=0)   
#
#class QuantifierTree(BaseEstimator, ClassifierMixin):
#    def __init__(self, max_depth=3, min_samples_leaf = 1, criterion="squared_error"):
#        self.classifier = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
#        self.max_depth = max_depth
#        self.min_samples_leaf = min_samples_leaf
#        self.criterion = criterion
#
#    def _fit(self, X, y, sample_weight=None, **kwargs):
#        return self.fit(X, y)
#         
#
#    def fit(self, X, y):
#        train_x, test_x, train_y, test_y = \
#            train_test_split(X, y, test_size=0.33, shuffle=True)
#        self.classifier.fit(train_x, train_y)
#        self.test_pred = self.classifier.predict_proba(test_x)[:,1]
#        self.pred_given_true = np.mean(self.test_pred[test_y.flatten() > 0])
#        self.pred_given_false = np.mean(self.test_pred[test_y.flatten() <= 0])
#        return self
#
#    def predict_(self, X):
#        return self.classifier.predict(X)
#    
#    def predict(self, X):
#        ypos_hat_chunk = self.classifier.predict_proba(X)[:,1]
#        p = (np.mean(ypos_hat_chunk) - self.pred_given_false) / (self.pred_given_true - self.pred_given_false)
#        if p < 0:
#            p = 0
#        elif p > 1:
#            p = 1
#        return p
#
#    
#    def _compute_missing_values_in_feature_mask(self, X, **kwargs):
#        # Compute the mask for missing values in the feature
#        return self.classifier._compute_missing_values_in_feature_mask(X, **kwargs)

class CNN(nn.Module):
    def __init__(self, input_size, num_filters, kernel_size=7, output_type = "sigmoid", epochs=200):
        self.epochs = epochs
        super(CNN, self).__init__()
        dilation = 5
        padding = "same"
        scaleup = .7
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(input_size, num_filters, kernel_size, stride=1, padding_mode="reflect", padding=padding,dilation=1)
        self.conv1b = nn.Conv1d(num_filters, num_filters, 1, stride=1, padding_mode="reflect")
        self.conv2 = nn.Conv1d(num_filters, int(num_filters *  (1+scaleup)), kernel_size, padding_mode="reflect", padding=padding,dilation=dilation)
        self.conv3 = nn.Conv1d(int(num_filters * (1+scaleup)), int(num_filters *  (1+scaleup*2)), kernel_size, padding_mode="reflect", padding=padding,dilation=dilation)
        self.conv3b = nn.Conv1d(int(num_filters *  (1+scaleup*2)), int(num_filters * (1+scaleup*3)), kernel_size, padding_mode="reflect", padding=padding,dilation=dilation)
        self.conv4 = nn.Conv1d(int(num_filters * (1+scaleup*3)), int( num_filters*(1+scaleup*4)), kernel_size, padding_mode="reflect", padding=padding,dilation=dilation)
        self.conv4b = nn.Conv1d(int(num_filters * (1+scaleup*4)) + input_size, int(num_filters * (1+scaleup*4)) + input_size, 1, stride=1, padding_mode="reflect")

        self.convOut = nn.Conv1d(int(num_filters * (1+scaleup*4)) + input_size, 1, kernel_size, padding_mode="reflect", padding=padding)
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool1d(kernel_size=dilation, padding=dilation//2, count_include_pad =False, stride=1)
        self.sigmoid = nn.Sigmoid() if output_type == "sigmoid" else nn.Identity()


    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv1b(out)
        out = self.pool(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.pool(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.pool(out)
        out = self.relu(out)
        out = self.conv3b(out)
        out = self.pool(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.relu(out)
        out = torch.cat((out, x), dim=0)
        out = self.conv4b(out)
        out = self.relu(out)
        out = self.convOut(out)
        return self.sigmoid(out)
    
    def fit(self, X, y,n_epoch = 200):
        # Convert to PyTorch tensors
        print(X.shape)
        dataset = torch.utils.data.TensorDataset(
                    torch.tensor(X, dtype=torch.float32), 
                    torch.tensor(y, dtype=torch.float32))
        
        batch_size = 6800
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        i_val = []
        for _, y_batch in dataloader:
            i_val.append(np.random.choice(y_batch.shape[0]-1000, int((y_batch.shape[0]-1000)*0.1), replace=False) + 1000)
            print(y_batch.shape)

        print(np.sum((X < 0)[:]))
        print(X)
        print(f"Num negative examples {np.sum((X < 0)[:])}",flush=True)

        # Define loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.RAdam(self.parameters())

        self.train()
        # Training loop
        epochs_since_best_loss = 0
        min_val_loss = np.inf
        for epoch in range(self.epochs):
            epoch_loss = 0
            epoch_loss_val = 0
            i = 0
            offset = np.random.randint(0, 1000)
            dataset = torch.utils.data.TensorDataset(
                torch.tensor(X[offset:,:], dtype=torch.float32), 
                torch.tensor(y[offset:], dtype=torch.float32))
        
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

            for i_batch, (x_batch, y_batch) in enumerate(dataloader):
                train_mask = torch.ones(y_batch.shape[0], dtype=torch.bool)
                train_mask[i_val[i_batch] - offset] = False
                optimizer.zero_grad()
                outputs = self(torch.swapaxes(x_batch, 0, 1))
                i_val[i_batch]
                loss = criterion(torch.flatten(outputs)[train_mask], y_batch[train_mask])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                loss_val = criterion(torch.flatten(outputs)[~train_mask], y_batch[~train_mask])
                epoch_loss_val += loss_val.item()
                i += 1
            if epoch_loss_val < min_val_loss:
                min_val_loss = epoch_loss_val
                epochs_since_best_loss = 0
                torch.save(self.state_dict, 'my_model_best_loss.pth')
            else:
                epochs_since_best_loss += 1
            print(f"Epoch {epoch+1}/{n_epoch}, Loss: {epoch_loss/i}, Val Loss: {epoch_loss_val/i}")

        self.load_state_dict(torch.load("my_model_best_loss.pth")())
        

    def predict(self, X):
        return torch.flatten((self(torch.swapaxes(torch.tensor(X, dtype=torch.float32), 0, 1)))).detach().numpy()
        
class CNNSpace(nn.Module):
    def __init__(self, input_size, num_filters, kernel_size=7):
        super().__init__()
        dilation = 1
        stride = kernel_size
        self.kernel_size = kernel_size
        padding = "valid"
        scaleup = 1
        self.conv1 = nn.Conv1d(input_size, num_filters, kernel_size, 
                               stride=stride, padding_mode="reflect", padding=padding,dilation=1)
        self.conv2 = nn.Conv1d(num_filters, int(num_filters *  (1+scaleup)), kernel_size, padding_mode="reflect", 
                               stride=stride, padding=padding,dilation=dilation)
        self.conv3 = nn.Conv1d(int(num_filters * (1+scaleup)), int(num_filters *  (1+scaleup*2)), kernel_size, 
                               stride=stride, padding_mode="reflect", padding=padding,dilation=dilation)
        self.conv4 = nn.Conv1d(int(num_filters * (1+scaleup*2)), num_filters*(1+scaleup*3), kernel_size, padding_mode="reflect",
                               stride=1, padding="same",dilation=dilation)
        self.conv4b = nn.Conv1d(int(num_filters * (1+scaleup*3)) + input_size, 
                                int(num_filters * (1+scaleup*4)) + input_size,
                                  1, stride=1, padding_mode="reflect")

        self.convOut = nn.Conv1d(int(num_filters * (1+scaleup*4)) + input_size, 1, kernel_size, padding_mode="reflect", padding="same", dilation=kernel_size**3)
        self.deconv1  = nn.ConvTranspose1d(kernel_size=kernel_size, 
                                          in_channels=int(num_filters * (1+scaleup*3)), 
                                          out_channels=int(num_filters * (1+scaleup*3)), 
                                          stride=kernel_size)
        self.deconv2  = nn.ConvTranspose1d(kernel_size=kernel_size, 
                                    in_channels=int(num_filters * (1+scaleup*3)), 
                                    out_channels=int(num_filters * (1+scaleup*3)), 
                                    stride=kernel_size)
        self.deconv3  = nn.ConvTranspose1d(kernel_size=kernel_size, 
                                in_channels=int(num_filters * (1+scaleup*3)), 
                                out_channels=int(num_filters * (1+scaleup*3)), 
                                stride=kernel_size)
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool1d(kernel_size=kernel_size, padding=kernel_size//2, count_include_pad =False, stride=kernel_size )
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        out = F.pad(x, (0, self.kernel_size - x.shape[-1]%self.kernel_size), mode='reflect')
        out = self.conv1(out)
        #print(f"Input shape: {x.shape}, Conv1 output shape: {out.shape}")
        out = self.relu(out)
        #print(out.shape, out.shape[-1]%self.kernel_size)
        out = F.pad(out, (0, self.kernel_size - out.shape[-1]%self.kernel_size), mode='reflect')
        out = self.conv2(out)
        out = self.relu(out)
        #print(out.shape, out.shape[-1]%self.kernel_size)
        out = F.pad(out, (0, self.kernel_size - out.shape[-1]%self.kernel_size), mode='reflect')
        out = self.conv3(out)
        out = self.relu(out)
        #print(out.shape, out.shape[-1]%self.kernel_size)
        out = self.conv4(out)
        out = self.relu(out)
        #print(out.shape)

        out = self.deconv1(out)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = out[:, :x.shape[-1]]  # Crop to original length
        #print(out.shape)
        out = torch.cat((out, x), dim=0)
        out = self.conv4b(out)
        out = self.relu(out)
        out = self.convOut(out)
        return self.sigmoid(out)
    
    def fit(self, X, y):
        # Convert to PyTorch tensors
        dataset = torch.utils.data.TensorDataset(
                    torch.tensor(X, dtype=torch.float32), 
                    torch.tensor(y, dtype=torch.float32))
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=12800, shuffle=False)

        # Define loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0002)

        self.train()
        # Training loop
        for epoch in range(300):
            for x_batch, y_batch in dataloader:
                optimizer.zero_grad()
                outputs = self(torch.swapaxes(x_batch, 0, 1))
                
                loss = criterion(torch.flatten(outputs), y_batch)
                loss.backward()
                optimizer.step()
                print(y_batch.shape, outputs.shape, loss.item())

    def predict(self, X):
        return torch.flatten((self(torch.swapaxes(torch.tensor(X, dtype=torch.float32), 0, 1)))).detach().numpy()
    
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1])
        return out
    
    def fit(self, X, y):
        # Convert to PyTorch tensors
        dataset = torch.utils.data.TensorDataset(
                    torch.tensor(X, dtype=torch.float32), 
                    torch.tensor(y, dtype=torch.float32))
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

        # Define loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        self.train()
        # Training loop
        for epoch in range(100):
            for x_batch, y_batch in dataloader:
                optimizer.zero_grad()
                print(x_batch.shape)
                outputs = self(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
        

class Model:
    def __init__(self, dataset_name=None):
        # Choose model based on dataset
        self.dataset_name = dataset_name
        self.regressor2 = RandomForestRegressor(n_estimators=1000, max_depth=10, min_samples_leaf=100)
        if dataset_name == 'dataset1':
            self.file = "CTR_dataset_modified.csv"
            self.target_column = "TARGET_CTR"
            self.regressor = CNN(4,12, kernel_size=7)#GradientBoostingRegressor( n_estimators=1000, max_depth=10,  min_samples_leaf=100)
            #self.classifier = GradientBoostingClassifier( n_estimators=1000, max_depth=10,  min_samples_leaf=1000)
            #self.quantifier = RandomForestQuantifier(n_estimators=1000, max_depth=10, min_samples_leaf=100)
        elif dataset_name == 'dataset2':
            self.file = "CONV_dataset_modified.csv"
            self.target_column = "TARGET_CONV"
            self.regressor = CNN(4,12, kernel_size=7, output_type="Linear", epochs=100)
            #GradientBoostingRegressor( n_estimators=1000, max_depth=10,  min_samples_leaf=100)
            #self.classifier = GradientBoostingClassifier( n_estimators=1000, max_depth=10,  min_samples_leaf=100, class_weight="balanced")
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    @staticmethod
    def derived_features(X):
        X_derived = (X[:,[0]] / X[:,[1]])
        X_derived[np.isinf(X_derived)] = 0
        X_derived[np.isnan(X_derived)] = 0
        #X_lead = np.concatenate((X[1:,:], 0*X[:1,:]))
        #X_lag = np.concatenate((0*X[-1:,:], X[:-1,:]))
        return X_derived#np.concatenate((X_derived, X_lead, X_lag), axis=1)


    def fit(self, _X, _y, override = True):
        #X = np.concatenate((X, self.derived_features(X)), axis=1)
        if override:
            df = pd.read_csv(filepath / self.file)
            X = df.drop(columns=[self.target_column]).fillna(0)
            y = df[self.target_column]
            y.fillna(0, inplace=True)  # Fill NaN values with 0
            X = X.values
            y = y.values
        else:
            X = _X
            y = _y
        X[X<0] = 0
        self.X = X
        self.y = y
        is_na_X = np.any(np.isnan(X), axis=1)
        is_na_y = np.isnan(y)
        is_na = is_na_X | is_na_y
        if is_na.any():
            X = X[~is_na,:]
            y = y[~is_na]
        print(f"Num negative examples {np.sum((X < 0)[:])}",flush=True)
        self.regressor.fit(X, y)
       # self.classifier.fit(X, y > 0)
       # self.quantifier.fit(X, y > 0)
       # train_pred = self.classifier.predict_proba(X)[:,1]
       # print(f"Classifier score: {roc_auc_score(y>0, train_pred)}")
       # self.train_p = np.mean(y > 0)
       # self.pred_given_true = np.mean(train_pred[y>0])
       # self.pred_given_false = np.mean(train_pred[y<=0])
        #self.regressor2.fit(X, y)

    def predict_prev(self, X):
        ypos_hat_chunk = self.classifier.predict_proba(X)[:,1]
        p = (np.mean(ypos_hat_chunk) - self.pred_given_false) / (self.pred_given_true - self.pred_given_false)
        if p < 0:
            p = 0
        elif p > 1:
            p = 1
        return p

    def predict_prev_em(self, X):
        p_em = self.train_p
        ypos_hat_chunk = self.classifier.predict_proba(X)[:,1]
        for epoch in range(100):
            posterior = (ypos_hat_chunk * p_em / self.train_p)/ \
                    (ypos_hat_chunk * p_em / self.train_p + (1 - ypos_hat_chunk) * (1 - p_em) / (1 - self.train_p))
            p_em = np.mean(posterior)
        return p_em


    def predict(self, X):
        posteriors = []
       # np.save(filepath / f"X_{self.dataset_name}_test.npy", X)
        X[np.isnan(X)] = 0
        X[X<0] = 0
        out = self.regressor.predict(X)
        #X = np.concatenate((X, self.derived_features(X)), axis=1)
        #
        #for i_slice in gen_batches(X.shape[0], 5000):
        #    X_chunk = X[i_slice,:]
        #    if X_chunk.shape[0] == 0:
        #        continue
        #    p = self.quantifier.predict(X_chunk)
        #    ypos_hat_chunk = self.classifier.predict_proba(X_chunk)[:,1]
        #    print(f"p: {p}")
        #    posterior = (ypos_hat_chunk * p / self.train_p)/ \
        #                (ypos_hat_chunk * p / self.train_p + (1 - ypos_hat_chunk) * (1 - p) / (1 - self.train_p))
        #    posteriors.append(posterior)
        #posteriors = np.concatenate(posteriors)
        #out_given_true = self.regressor.predict(X)
        out[out<0] = 0
        return out
