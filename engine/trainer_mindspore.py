import mindspore.nn as nn
from mindspore.train import Model
from mindvision.engine.callback import ValAccMonitor

class Trainer:
    def __init__(self, model, config, train_dloader, val_dloader, test_dloader):
        self.model = model
        self.config = config
        self.train_dloader = train_dloader
        self.val_dloader = val_dloader
        self.test_dloader = test_dloader
        self.loss_func = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        self.optimizer = nn.Momentum(self.model.trainable_params(), learning_rate=config['learning_rate'], momentum=config['momentum'])

    def train(self):
        self.model = Model(self.model, loss_fn=self.loss_func, optimizer=self.optimizer, metrics={'accuracy'})
        self.model.train(self.config['epoch_num'], self.train_dloader, callbacks=[ValAccMonitor(self.model, self.val_dloader, num_epochs=1)])

    def test(self):
        acc = self.model.eval(self.test_dloader)
        return acc
