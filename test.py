import needle as ndl
import numpy as np
from needle import nn
from matplotlib import pyplot as plt

A = np.array([[1, 2], [-0.2, 0.5]])
mu = np.array([2, 1])
# total number of sample data to generated
num_sample = 3200
data = np.random.normal(0, 1, (num_sample, 2)) @ A + mu


# plt.scatter(data[:,0], data[:,1], color="blue", label="real data")
# plt.show()

model_G = nn.Sequential(nn.Linear(2, 2))
num_samples = 3200

def sample_G(model_G, num_samples):
    Z = ndl.Tensor(np.random.normal(0, 1, (num_samples, 2)).astype("float32"))
    return model_G(Z).numpy()

fake_data_init = sample_G(model_G, num_samples)

# plt.scatter(data[:,0], data[:,1], color="blue", label="real data")
# plt.scatter(fake_data_init[:, 0], fake_data_init[:, 1], color="pink", label="init_G")
# plt.show()

# model_D = nn.Sequential(
#         nn.Linear(2, 10),
#         nn.ReLU(),
#         nn.Linear(10, 20),
#         nn.ReLU(),
#         nn.Linear(20, 2)
#         )
# loss_D = nn.SoftmaxLoss()

# opt_G = ndl.optim.Adam(model_G.parameters(), lr=0.02)

# def update_G(Z, model_G, model_D, opt_G):
#     fakeX = model_G(Z)
#     fakeY = model_D(fakeX)
#     batch_size = Z.shape[0]
#     ones = ndl.ones(batch_size, dtype="int32")
#     loss = loss_D(fakeY, ones)
#     loss.backward()
#     opt_G.step()

# opt_D = ndl.optim.Adam(model_D.parameters(), lr=0.02)

# def update_D(Z, X, model_G, model_D, opt_D):
#     fakeX = model_G(Z).detach()
#     fakeY = model_D(fakeX)
#     realY = model_D(X)
#     batch_size = Z.shape[0]
#     ones = ndl.ones(batch_size, dtype="int32")
#     zeros = ndl.zeros(batch_size, dtype="int32")
#     loss = loss_D(fakeY, zeros) + loss_D(realY, ones)
#     loss.backward()
#     opt_D.step()

# def train_gan(data, batch_size, num_epochs):
#     assert data.shape[0] % batch_size == 0

#     for epoch in range(num_epochs):
#         begin = batch_size * epoch % data.shape[0]
#         X = ndl.Tensor(data[begin: begin + batch_size], dtype="float32")
#         Z = ndl.Tensor(np.random.normal(0, 1, (batch_size, 2)), dtype="float32")

#         update_G(Z, model_G, model_D, opt_G)
#         update_D(Z, X, model_G, model_D, opt_D)

# train_gan(data, 32, 1000)
# fake_data_trained = sample_G(model_G, num_samples)

# plt.scatter(data[:,0], data[:,1], color="blue", label="real data")
# plt.scatter(fake_data_init[:, 0], fake_data_init[:, 1], color="pink", label="init_G")
# plt.scatter(fake_data_trained[:, 0], fake_data_trained[:, 1], color="red", label="train_G")
# plt.legend()
# plt.show()

# gA, gmu = model_G.parameters()
# print(A.T @ A)
# gA = gA.numpy()
# print(gA.T @ gA)
# print(gmu, mu)

class GANLoss:
    def __init__(self, model_D, opt_D) -> None:
        self.model_D = model_D
        self.opt_D = opt_D
        self.loss_D = nn.SoftmaxLoss()

    def _update_D(self, fakeX, realX):
        fakeY = self.model_D(fakeX)
        realY = self.model_D(realX)
        batch_size = fakeX.shape[0]
        ones = ndl.ones(batch_size, dtype="int32")
        zeros = ndl.zeros(batch_size, dtype="int32")
        loss = self.loss_D(fakeY, zeros) + self.loss_D(realY, ones)
        loss.backward()
        self.opt_D.step()

    def forward(self, fakeX, realX):
        self._update_D(fakeX, realX)
        fakeY = self.model_D(fakeX)
        batch_size = fakeX.shape[0]

        ones = ndl.ones(batch_size, dtype="int32")
        loss = self.loss_D(fakeY, ones)
        return loss

model_G = nn.Sequential(nn.Linear(2,2))
opt_G = ndl.optim.Adam(model_G.parameters(), lr=0.01)

model_D = nn.Sequential(
        nn.Linear(2,10),
        nn.ReLU(),
        nn.Linear(10,20),
        nn.ReLU(),
        nn.Linear(20,2)
        )
opt_D = ndl.optim.Adam(model_D.parameters(), lr=0.01)

gan_loss = GANLoss(model_D, opt_D)

def train_gan(data, batch_size, num_epochs):
    assert data.shape[0] % batch_size == 0
    for epoch in range(num_epochs):
        begin = batch_size * epoch % data.shape[0]
        X = ndl.Tensor(data[begin: begin + batch_size, :], dtype="float32")
        Z = ndl.Tensor(np.random.normal(0, 1, (batch_size, 2)), dtype="float32")
        fakeX = model_G(Z)
        loss = gan_loss.forward(fakeX, X)
        loss.backward()
        opt_G.step()

train_gan(data, 32, num_samples)
fake_data_trained = sample_G(model_G, num_samples)

plt.scatter(data[:,0], data[:,1], color="blue", label="real data")
plt.scatter(fake_data_init[:, 0], fake_data_init[:, 1], color="pink", label="init_G")
plt.scatter(fake_data_trained[:, 0], fake_data_trained[:, 1], color="red", label="train_G")
plt.legend()
plt.show()
