class Trainable:

    def __init__(self, dataset):
        self.dataset = dataset

    def train(self, epochs, batch_size=8):

        loader = DataLoader(self.dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=4)

        for epoch in range(epochs):
            for i, batch in enumerate(dataset_loader):
                global_step = i + epoch * len(dataset_loader)
                self.do_batch(batch, global_step)
            self.do_epoch()
