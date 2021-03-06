from tqdm import tqdm
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.parameter import ParameterTuple

class LossFunc(nn.LossBase):
    def __init__(self, reduction='mean'):
        super(LossFunc, self).__init__(reduction)
        self.bce = nn.BCELoss(reduction='mean')

    def construct(self, logits, label):
        loss = self.bce(logits, label)
        return self.get_loss(loss) * label.shape[1]

def AccFunc(out, ans):
    sparse_outs = ops.Argmax(1)(out)
    one_hot = nn.OneHot(depth = 457)
    sparse_outs = one_hot(sparse_outs)
    corrects = ops.ReduceSum()(ans * sparse_outs, 1)
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
        train_cell = TrainCell(self.model, self.config)
        train_cell.set_train(True)
        eval_cell = EvalCell(self.model)
        eval_cell.set_train(False)
        for i in range(self.config['epoch_num']):
            iterator = self.train_dloader.create_dict_iterator()
            for item in tqdm(iterator, desc='Train epoch{:3d}'.format(i), ncols=0, total=self.train_dloader.get_dataset_size()):
                img = item['img']
                que = item['que']
                ans = item['ans']
                train_cell(img, que, ans)
            sum_loss = 0
            sum_acc = 0
            num = 0
            iterator = self.val_dloader.create_dict_iterator()
            for item in tqdm(iterator, desc='Validate epoch{:3d}'.format(i), ncols=0, total=self.val_dloader.get_dataset_size()):
                img = item['img']
                que = item['que']
                ans = item['ans']
                loss, acc = eval_cell(img, que, ans)
                sum_loss += loss
                sum_acc += acc
                num += 1
            print('Epoch{:3d}:'.format(i))
            print('Loss: {:.4f}, Accuracy: {:.4f}'.format(float(sum_loss.asnumpy()) / num, float(sum_acc.asnumpy()) / num))

    def test(self):
        eval_cell = EvalCell(self.model)
        eval_cell.set_train(False)
        sum_loss = 0
        sum_acc = 0
        num = 0
        iterator = self.test_dloader.create_dict_iterator()
        for item in tqdm(iterator, desc='Test', ncols=0, total=self.test_dloader.get_dataset_size()):
            img = item['img']
            que = item['que']
            ans = item['ans']
            loss, acc = eval_cell(img, que, ans)
            sum_loss += loss
            sum_acc += acc
            num += 1
        print('Test results:')
        print('Loss: {:.4f}, Accuracy: {:.4f}'.format(float(sum_loss.asnumpy()) / num, float(sum_acc.asnumpy()) / num))
