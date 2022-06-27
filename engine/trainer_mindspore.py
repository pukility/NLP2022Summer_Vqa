import tqdm
import math
import mindspore.nn as nn
from mindspore.nn.loss.loss import _Loss
from mindspore.ops import operations as ops
from mindspore.common.parameter import ParameterTuple

class LossFunc(_Loss):
    def __init__(self, reduction='mean'):
        super(LossFunc, self).__init__(reduction)
        self.reduce_sum = ops.ReduceSum()
        self.log_softmax = ops.LogSoftmax(axis=0)

    def construct(self, logits, label):
        nll = -self.log_softmax(logits)
        loss = self.reduce_sum(nll * label / 10, axis=1).mean()
        return self.get_loss(loss)

def AccFunc(out, ans):
    arg_max = ops.Argmax(1)
    gather = ops.GatherD()
    minimum = ops.Minimum()
    unsqueeze = ops.ExpandDims()
    squeeze = ops.Squeeze(1)

    predicted_index = arg_max(out)
    predicted_index = unsqueeze(predicted_index, 1)
    agreeing = gather(ans, 1, predicted_index)
    agreeing = squeeze(agreeing)
    return minimum(agreeing * 0.3, 1.0)

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
        optimizer = nn.Adam(self.model.trainable_params(), learning_rate=config['learning_rate'])
        self.loss_train_net = TrainOneStepCell(loss_net, optimizer)

    def construct(self, img, que, ans):
        loss = self.loss_train_net(img, que, ans)
        return loss

class EvalCell(nn.Cell):
    def __init__(self, net):
        super(EvalCell, self).__init__()
        self.net = net
        self._loss_fn = LossFunc()

    def construct(self, img, que, ans):
        out = self.net(img, que)
        loss = self._loss_fn(out, ans)
        acc = AccFunc(out, ans)
        return loss, acc

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
            for img, que, ans in tqdm(self.train_dloader, desc='Train epoch{:03d}'.format(i), ncols=0, total=math.ceil(len(self.train_dloader.source) / self.config['batch_size'])):
                train_cell(img, que, ans)
            sum_loss = 0
            sum_acc = 0
            num = 0
            for img, que, ans in tqdm(self.val_dloader, desc='Validate epoch{:03d}'.format(i), ncols=0, total=math.ceil(len(self.val_dloader.source) / self.config['batch_size'])):
                loss, acc = EvalCell(img, que, ans)
                sum_loss += loss
                sum_acc += acc
                num += 1
            print('Epoch{:03d}:'.format(i))
            print('Loss: {:.4f}, Accuracy: {:.4f}'.format(sum_loss, sum_acc / num))

    def test(self):
        sum_loss = 0
        sum_acc = 0
        num = 0
        for img, que, ans in tqdm(self.test_dloader, desc='Test', ncols=0, total=math.ceil(len(self.test_dloader.source) / self.config['batch_size'])):
            loss, acc = EvalCell(img, que, ans)
            sum_loss += loss
            sum_acc += acc
            num += 1
        print('Test results:')
        print('Loss: {:.4f}, Accuracy: {:.4f}'.format(sum_loss, sum_acc / num))
