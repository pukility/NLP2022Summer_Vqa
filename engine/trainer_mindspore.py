from tqdm import tqdm
import mindspore.nn as nn
import mindspore.numpy as np
import mindspore.ops as ops
from mindspore.common.parameter import ParameterTuple

class LossFunc(nn.LossBase):
    def __init__(self, reduction='mean'):
        super(LossFunc, self).__init__(reduction)
        self.ce = nn.SoftmaxCrossEntropyWithLogits(sparse=False)

    def construct(self, logits, label):
        loss = self.ce(logits, label)
        return self.get_loss(loss)

def AccFunc(out, ans):
    labels = ops.Argmax(1)(ans)
    sparse_labels = ops.ZerosLike()(ans)
    for i in range(ops.Shape()(ans)[0]):
        sparse_labels[i][labels[i]] = 1
    sparse_outs = np.where(out > 1, 1, 0)
    corrects = ops.ReduceSum()(sparse_labels * sparse_outs, 1)
    return ops.ReduceMean()(corrects, 0)

class TrainOneStepCell(nn.Cell):
    def __init__(self, loss_net, optimizer):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.loss_net = loss_net
        self.loss_net.add_flags(defer_inline=True)
        self.weights = ParameterTuple(loss_net.trainable_params())
        self.optimizer = optimizer
        self.grad = ops.GradOperation(get_by_list=True)

    def construct(self, img, que, ans):
        weights = self.weights
        loss = self.loss_net(img, que, ans)
        grads = self.grad(self.loss_net, weights)(img, que, ans)
        return ops.depend(loss, self.optimizer(grads))

class TrainLossCell(nn.Cell):
    def __init__(self, net):
        super(TrainLossCell, self).__init__(auto_prefix=False)
        self.loss_fn = LossFunc()
        self.net = net

    def construct(self, img, que, ans):
        out = self.net(img, que)
        loss = self.loss_fn(out, ans)
        return loss

class TrainCell(nn.Cell):
    def __init__(self, net, config):
        super(TrainCell, self).__init__(auto_prefix=False)
        loss_net = TrainLossCell(net)
        optimizer = nn.Adam(loss_net.trainable_params(), learning_rate=config['learning_rate'])
        self.loss_train_net = TrainOneStepCell(loss_net, optimizer)

    def construct(self, img, que, ans):
        loss = self.loss_train_net(img, que, ans)
        return loss.asnumpy()

class EvalCell(nn.Cell):
    def __init__(self, net):
        super(EvalCell, self).__init__()
        self.net = net
        self._loss_fn = LossFunc()

    def construct(self, img, que, ans):
        out = self.net(img, que)
        loss = self._loss_fn(out, ans)
        acc = AccFunc(out, ans)
        return loss.asnumpy(), acc.asnumpy()

class Trainer:
    def __init__(self, model, config, train_dloader, val_dloader, test_dloader):
        self.model = model
        self.config = config
        self.train_dloader = train_dloader
        self.val_dloader = val_dloader
        self.test_dloader = test_dloader

    def train(self):
        # self.model = Model(self.model, loss_fn=self.loss_func, optimizer=self.optimizer, metrics={'accuracy'})
        # self.model.train(self.config['epoch_num'], self.train_dloader, callbacks=[ValAccMonitor(self.model, self.val_dloader, num_epochs=1)])
        train_cell = TrainCell(self.model, self.config)
        train_cell.set_train(True)
        eval_cell = EvalCell(self.model)
        eval_cell.set_train(False)
        for i in range(self.config['epoch_num']):
            iterator = self.train_dloader.create_tuple_iterator()
            for img, que, ans in tqdm(iterator, desc='Train epoch{:3d}'.format(i), ncols=0, total=self.train_dloader.get_dataset_size()):
                train_cell(img, que, ans)
            sum_loss = 0
            sum_acc = 0
            num = 0
            iterator = self.val_dloader.create_tuple_iterator()
            for img, que, ans in tqdm(iterator, desc='Validate epoch{:3d}'.format(i), ncols=0, total=self.val_dloader.get_dataset_size()):
                loss, acc = eval_cell(img, que, ans)
                sum_loss += loss
                sum_acc += acc
                num += 1
            print('Epoch{:03d}:'.format(i))
            print('Loss: {:.4f}, Accuracy: {:.4f}'.format(sum_loss, sum_acc / num))

    def test(self):
        sum_loss = 0
        sum_acc = 0
        num = 0
        iterator = self.test_dloader.create_tuple_iterator()
        for img, que, ans in tqdm(iterator, desc='Test', ncols=0, total=self.test_dloader.get_dataset_size()):
            loss, acc = eval_cell(img, que, ans)
            sum_loss += loss
            sum_acc += acc
            num += 1
        print('Test results:')
        print('Loss: {:.4f}, Accuracy: {:.4f}'.format(sum_loss, sum_acc / num))
