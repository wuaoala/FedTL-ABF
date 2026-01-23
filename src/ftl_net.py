import torch
import numpy as np
np.random.seed(1)
class FTL_nn(torch.nn.Module):
    def __init__(self, models, optimizers, data_owner):

        self.data_owners = data_owner
        self.optimizers = optimizers
        self.models = models
        super().__init__()

    def forward(self, data_pointer, alignment):

        # individual client's output upto their respective cut layer
        client_output = {}

        # outputs that is moved to server and subjected to concatenate for server input
        remote_outputs = []
        feature_outputs = []
        # iterate over each client and pass their inputs to respective model segment and move outputs to server
        if alignment == 'non_aligned':
            self.data_owners = [self.data_owners[0]]
        for owner in self.data_owners:
            client_output[owner], client_fea_outputs = self.models[owner](data_pointer[owner])
            remote_outputs.append(client_output[owner].requires_grad_())
            feature_outputs.append(client_fea_outputs.requires_grad_())
        self.data_owners = ['client_1', 'client_2', 'client_3']
        return remote_outputs, feature_outputs


    def zero_grads(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def step(self):
        for opt in self.optimizers:
            opt.step()

    def train(self):
        for loc in self.models.keys():
            # for i in range(len(self.models[loc])):
            #     self.models[loc][i].train()
            self.models[loc].train()

    def eval(self):
        for loc in self.models.keys():
            for i in range(len(self.models[loc])):
                self.models[loc][i].eval()


