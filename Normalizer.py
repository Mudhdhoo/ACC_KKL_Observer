import torch

class Normalizer:
    def __init__(self, dataset):
        self.x_size = dataset.system.x_size
        self.z_size = dataset.system.z_size

        self.mean_x = dataset.mean_x
        self.std_x = dataset.std_x
        self.mean_z = dataset.mean_z
        self.std_z = dataset.std_z

        self.mean_x_ph = dataset.mean_x_ph
        self.std_x_ph = dataset.std_x_ph
        self.mean_z_ph = dataset.mean_z_ph
        self.std_z_ph = dataset.std_z_ph

        self.sys = dataset.system

    def check_sys(self, tensor, mode):
        """
        Checks if the tensor is x or z data. Then if the tensor belongs to the 
        physics or normal dataset. The correct mean and standard deviations are chosen 
        according to those parameters.
        """
        if tensor.size()[1] == self.sys.x_size:     # Check if x or z input
            if mode == 'physics':       # Check if physics or normal data point
                mean = self.mean_x_ph
                std = self.std_x
            else:
                mean = self.mean_x
                std = self.std_x
        elif tensor.size()[1] == self.sys.z_size:
            if mode == 'physics':
                mean = self.mean_z_ph
                std = self.std_z_ph
            else:
                mean = self.mean_z
                std = self.std_z
        else:
            raise Exception('Size of tensor unmatched with any system.')     

        return mean, std

    def Normalize(self, tensor, mode):
        mean, std = self.check_sys(tensor, mode)            
        return (tensor - mean) / std

    def Denormalize(self, tensor, mode):
        mean, std = self.check_sys(tensor, mode)           
        return tensor*std + mean        