import numpy as np
import torch


class Distribute_Data:
    """
  This class distribute each image among different workers
  It returns a dictionary with key as data owner's id and
  value as a pointer to the list of data batches at owner's
  location.

  example:-
  >>> from distribute_data import Distribute_MNIST
  >>> obj = Distribute_MNIST(data_owners= (alice, bob, claire), data_loader= torch.utils.data.DataLoader(trainset))
  >>> obj.data_pointer[1]['alice'].shape, obj.data_pointer[1]['bob'].shape, obj.data_pointer[1]['claire'].shape
   (torch.Size([64, 1, 9, 28]),
    torch.Size([64, 1, 9, 28]),
    torch.Size([64, 1, 10, 28]))
  """

    def __init__(self, data_owners, data_loader,data_name, alignment,data_from):

        """
         Args:
          data_owners: tuple of data owners
          data_loader: torch.utils.data.DataLoader for MNIST
        """

        self.data_owners = data_owners
        self.data_loader = data_loader
        self.no_of_owner = len(data_owners)

        self.data_pointer = []
        """
        self.data_pointer:  list of dictionaries where 
        (key, value) = (id of the data holder, a pointer to the list of batches at that data holder).
        example:
        self.data_pointer  = [
                                {"alice": pointer_to_alice_batch1, "bob": pointer_to_bob_batch1},
                                {"alice": pointer_to_alice_batch2, "bob": pointer_to_bob_batch2},
                                ...
                             ]
        """

        self.labels = []
        # iterate over each batch of dataloader for, 1) spliting image 2) sending to VirtualWorker
        for images, labels in self.data_loader:
            curr_data_dict = {}
            # calculate width and height according to the no. of workers for UNIFORM distribution
            height = images.shape[-1] // self.no_of_owner
            self.labels.append(labels)
            if data_name == 'Taiwan':
                if alignment == 'aligned':
                    curr_data_dict['client_1'] = images[:, : 16]
                    curr_data_dict['client_2'] = images[:, 16: 49]
                    curr_data_dict['client_3'] = images[:, 49:]
                else:
                    if data_from == "train":
                        curr_data_dict['client_1'] = images[:6400, : 16]
                        curr_data_dict['client_2'] = images[6400:12800, 16: 49]
                        curr_data_dict['client_3'] = images[12800:, 49:]
                        # 非对齐测试样本划分
                    else:
                        curr_data_dict['client_1'] = images[:, : 16]
                        curr_data_dict['client_2'] = images[1600:3200, 16: 49]
                        curr_data_dict['client_3'] = images[3200:, 49:]
            elif data_name == 'GMSC':
                if alignment == 'aligned':
                    curr_data_dict['client_1'] = images[:, : 2]
                    curr_data_dict['client_2'] = images[:, 2: 6]
                    curr_data_dict['client_3'] = images[:, 6:]
                else:
                    if data_from == "train":
                        curr_data_dict['client_1'] = images[:32000, : 2]
                        curr_data_dict['client_2'] = images[32000:64000, 2: 6]
                        curr_data_dict['client_3'] = images[64000:, 6:]
                        # 非对齐测试样本划分
                    else:
                        curr_data_dict['client_1'] = images[:, : 2]
                        curr_data_dict['client_2'] = images[8000:16000, 2: 6]
                        curr_data_dict['client_3'] = images[16000:, 6:]
            elif data_name == 'LD':
                if alignment == 'aligned':
                    curr_data_dict['client_1'] = images[:, : 4]
                    curr_data_dict['client_2'] = images[:, 4: 12]
                    curr_data_dict['client_3'] = images[:, 12:]
                else:
                    if data_from == "train":
                        curr_data_dict['client_1'] = images[:262, : 4]
                        curr_data_dict['client_2'] = images[262:523, 4: 12]
                        curr_data_dict['client_3'] = images[523:, 12:]
                        # 非对齐测试样本划分
                    else:
                        curr_data_dict['client_1'] = images[:, : 4]
                        curr_data_dict['client_2'] = images[66:131, 4: 12]
                        curr_data_dict['client_3'] = images[131:, 12:]
            elif data_name == 'HMEQ':
                if alignment == 'aligned':
                    curr_data_dict['client_1'] = images[:, : 14]
                    curr_data_dict['client_2'] = images[:, 14: 39]
                    curr_data_dict['client_3'] = images[:, 39:]
                else:
                    if data_from == "train":
                        curr_data_dict['client_1'] = images[:1272, : 14]
                        curr_data_dict['client_2'] = images[1272:2544, 14: 39]
                        curr_data_dict['client_3'] = images[2544:, 39:]
                        # 非对齐测试样本划分
                    else:
                        curr_data_dict['client_1'] = images[:, : 14]
                        curr_data_dict['client_2'] = images[318:636, 14: 39]
                        curr_data_dict['client_3'] = images[636:, 39:]
            elif data_name == 'German':
                if alignment == 'aligned':
                    curr_data_dict['client_1'] = images[:, : 7]
                    curr_data_dict['client_2'] = images[:, 7: 32]
                    curr_data_dict['client_3'] = images[:, 32:]
                else:
                    if data_from == "train":
                        curr_data_dict['client_1'] = images[:214, : 7]
                        curr_data_dict['client_2'] = images[214:428, 7: 32]
                        curr_data_dict['client_3'] = images[428:, 32:]
                        # 非对齐测试样本划分
                    else:
                        curr_data_dict['client_1'] = images[:, : 7]
                        curr_data_dict['client_2'] = images[54:108, 7: 32]
                        curr_data_dict['client_3'] = images[108:, 32:]
            elif data_name == 'PAKDD':
                if alignment == 'aligned':
                    curr_data_dict['client_1'] = images[:, : 4]
                    curr_data_dict['client_2'] = images[:, 4: 11]
                    curr_data_dict['client_3'] = images[:, 11:]
                else:
                    if data_from == "train":
                        curr_data_dict['client_1'] = images[:8307, : 4]
                        curr_data_dict['client_2'] = images[8307:16614, 4: 11]
                        curr_data_dict['client_3'] = images[16614:, 11:]
                        # 非对齐测试样本划分
                    else:
                        curr_data_dict['client_1'] = images[:, : 4]
                        curr_data_dict['client_2'] = images[:, 4: 11]
                        curr_data_dict['client_3'] = images[:, 11:]
            elif data_name == 'LC':
                if alignment == 'aligned':
                    curr_data_dict['client_1'] = images[:, : 7]
                    curr_data_dict['client_2'] = images[:, 7: 19]
                    curr_data_dict['client_3'] = images[:, 19:]
                else:
                    if data_from == "train":
                        curr_data_dict['client_1'] = images[:9074, : 7]
                        curr_data_dict['client_2'] = images[9074:18148, 7: 19]
                        curr_data_dict['client_3'] = images[18148:, 19:]
                        # 非对齐测试样本划分
                    else:
                        curr_data_dict['client_1'] = images[:, : 7]
                        curr_data_dict['client_2'] = images[:, 7: 19]
                        curr_data_dict['client_3'] = images[:, 19:]
            elif data_name == 'HC':
                if alignment == 'aligned':
                    curr_data_dict['client_1'] = images[:, : 79]
                    curr_data_dict['client_2'] = images[:, 79: 199]
                    curr_data_dict['client_3'] = images[:, 199:]
                else:
                    if data_from == "train":
                        curr_data_dict['client_1'] = images[:65603, : 79]
                        curr_data_dict['client_2'] = images[65603:131205, 79: 199]
                        curr_data_dict['client_3'] = images[131205:, 199:]
                        # 非对齐测试样本划分
                    else:
                        curr_data_dict['client_1'] = images[:, : 79]
                        curr_data_dict['client_2'] = images[:, 79: 199]
                        curr_data_dict['client_3'] = images[:, 199:]


            elif data_name == 'Ant':
                if alignment == 'aligned':
                    curr_data_dict['client_1'] = images[:, : 9]
                    curr_data_dict['client_2'] = images[:, 9: 19]
                    curr_data_dict['client_3'] = images[:, 19:]
                else:
                    if data_from == "train":
                        curr_data_dict['client_1'] = images[:71744, : 9]
                        curr_data_dict['client_2'] = images[71744:143488, 9: 19]
                        curr_data_dict['client_3'] = images[143488:, 19:]
                    # 非对齐测试样本划分
                    else:
                        curr_data_dict['client_1'] = images[:, : 9]
                        curr_data_dict['client_2'] = images[:, 9: 19]
                        curr_data_dict['client_3'] = images[:, 19:]
            self.data_pointer.append(curr_data_dict)

    def __iter__(self):
        # for data_ptr, label in zip(self.data_pointer[:-1], self.labels[:-1]):
        for data_ptr, label in zip(self.data_pointer, self.labels):
            yield (data_ptr, label)

    def __len__(self):

        # return len(self.data_loader) - 1
        return len(self.data_loader)
