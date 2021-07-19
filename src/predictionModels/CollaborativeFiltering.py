import torch 
import numpy as np
from ..models.Data import Data

class PMFLoss(torch.nn.Module):
    def __init__(self, lam_u=0.3, lam_v=0.3):
        super().__init__()
        self.lam_u = lam_u
        self.lam_v = lam_v
    
    def forward(self, matrix, u_features, v_features):
        non_zero_mask = (matrix != -1).type(torch.FloatTensor)
        predicted = torch.sigmoid(torch.mm(u_features, v_features.t()))
        
        diff = (matrix - predicted)**2
        prediction_error = torch.sum(diff*non_zero_mask)

        u_regularization = self.lam_u * torch.sum(u_features.norm(dim=1))
        v_regularization = self.lam_v * torch.sum(v_features.norm(dim=1))
        
        return prediction_error + u_regularization + v_regularization

class CollaborativeFiltering:

    def __init__(self, data: Data):
        self.n_users, self.n_products = data.matrix.shape
        self.min_mv, self.max_mv = data.table['marketValue'].min(), data.table['marketValue'].max()

        self.values = data.matrix

        print(data.matrix.columns)

        for i in data.matrix.index:
            for j in data.matrix.columns:
                if data.matrix[j][i] - data.min[i] != 0:
                    data.matrix[j][i] = (data.matrix[j][i] - data.min[i])/(data.max[i] - data.min[i])
                else:
                    data.matrix[j][i] = 0.1
                
        data.matrix[data.matrix.isnull()] = -1

        data.matrix = torch.FloatTensor(data.matrix.values)

        self.table = data.table
        self.matrix = data.matrix

    def train(self, latent_vectors):
        self.user_features = torch.randn(self.n_users, latent_vectors, requires_grad=True)
        self.user_features.data.mul_(0.01)
        self.products_features = torch.randn(self.n_products, latent_vectors, requires_grad=True)
        self.products_features.data.mul_(0.01)

        pmferror = PMFLoss(lam_u=0.05, lam_v=0.05)
        optimizer = torch.optim.Adam([self.user_features, self.products_features], lr=0.01)
        for i in enumerate(range(1000)):
            optimizer.zero_grad()
            loss = pmferror(self.matrix, self.user_features, self.products_features)
            loss.backward()
            optimizer.step()

    def getPredictions(self):
        self.matrix = np.array(self.matrix)
        print(self.matrix)
        for i in range(self.n_users):
            predictions = torch.sigmoid(torch.mm(self.user_features[i, :].view(1, -1), self.products_features.t()))
            predictions = predictions.detach().numpy()
            for j in range(self.n_products):
                if self.matrix[i][j] == -1 and predictions[0][j] >= 0.6:
                    print('Recommend user ' + str(self.values.index[i]) + " product " + str(self.values.columns[j]))
        