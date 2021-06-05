import network as nt
import numpy as np

#net = nt.Network([784, 90, 30, 10])


eta = 0.0065
n_epochs = 60000

# for eta in np.linspace(init_eps, final_eps, 10):
train, test = nt.load_wrap_data()
net = nt.Network([784, 69, 28, 10])
net.transcribe_train_IGD_mod(train, eta, n_epochs, 0.92)
