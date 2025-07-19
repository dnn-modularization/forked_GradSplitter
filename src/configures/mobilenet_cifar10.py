from global_configure import GlobalConfigures


class Configures(GlobalConfigures):
    def __init__(self):
        super(Configures, self).__init__()
        self.model_name = 'mobilenet'
        self.dataset_name = 'cifar10'
        self.num_classes = 10

        self.workspace = f'{self.data_dir}/{self.model_name}_{self.dataset_name}'
