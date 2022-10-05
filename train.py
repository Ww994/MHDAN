from __future__ import print_function
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os
import math
import data_loader
import res_mix_adv as models

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
model_name='blc'
fw = open("result/print.txt", 'w')  # 保存路
# Training settings
batch_size = 32
iteration = 10000
lr = [0.001, 0.01]
momentum = 0.9
cuda = True
seed = 8
log_interval = 10
l2_decay = 5e-4
root_path = "../dataset/"
source1_name = "breast"
source2_name = "lung"
target_name = "colon"
# add
transfer_loss_weight =10.0
mix_rate=0.4
mix_ratio = 1.0
m_ratio_1=0.0001
m_ratio_2=0.0001
fw.write('mix_rate: {} \tmix_ratio: {}\tm_ratio_1: {}\tm_ratio_2:{}'.format(mix_rate,mix_ratio ,m_ratio_1,m_ratio_2))

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

source1_loader = data_loader.load_training(root_path, source1_name, batch_size, kwargs)
source2_loader = data_loader.load_training(root_path, source2_name, batch_size, kwargs)
target_train_loader = data_loader.load_training(root_path, target_name, batch_size, kwargs)
target_test_loader = data_loader.load_testing(root_path, target_name, batch_size, kwargs)


def train(model):
    source1_iter = iter(source1_loader)
    source2_iter = iter(source2_loader)
    target_iter = iter(target_train_loader)
    correct = 0
    acc_correct = 0
    optimizer = torch.optim.SGD([
        {'params': model.sharedNet.parameters()},
        {'params': model.cls_fc_son1.parameters(), 'lr': lr[1]},
        {'params': model.cls_fc_son2.parameters(), 'lr': lr[1]},
        {'params': model.sonnet1.parameters(), 'lr': lr[1]},
        {'params': model.sonnet2.parameters(), 'lr': lr[1]},
    ], lr=lr[0], momentum=momentum, weight_decay=l2_decay)

    for i in range(1, iteration + 1):
        model.train()

        optimizer.param_groups[0]['lr'] = lr[0] / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        optimizer.param_groups[1]['lr'] = lr[1] / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        optimizer.param_groups[2]['lr'] = lr[1] / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        optimizer.param_groups[3]['lr'] = lr[1] / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        optimizer.param_groups[4]['lr'] = lr[1] / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)

        try:
            source_data, source_label = source1_iter.next()
        except Exception as err:
            source1_iter = iter(source1_loader)
            source_data, source_label = source1_iter.next()
        try:
            target_data, target_label = target_iter.next()
        except Exception as err:
            target_iter = iter(target_train_loader)
            target_data, target_label = target_iter.next()
        if cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
            target_data = target_data.cuda()
        source_data, source_label = Variable(source_data), Variable(source_label)
        target_data = Variable(target_data)
        optimizer.zero_grad()
        # add
        cls_loss, mmd_loss, mmd_loss_m, l1_loss,mix_loss,advloss = model(source_data, target_data, source_label, mark=1,ratio=0.1)
        gamma = 2 / (1 + math.exp(-10 * (i) / (iteration))) - 1
        # add
        loss = cls_loss + gamma * (mmd_loss + l1_loss) + mix_ratio * mmd_loss_m+m_ratio_1*mix_loss
        loss = loss + transfer_loss_weight *advloss
        loss.backward()
        optimizer.step()

        if i % log_interval == 0:
            print(
                'Train source1 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}\tl1_Loss: {:.6f}\tmmd_loss_m: {:.6f}\tmix_Loss: {:.6f}'.format(
                    i, 100. * i / iteration, loss.item(), cls_loss.item(), mmd_loss.item(), l1_loss.item(),
                    mmd_loss_m.item(),mix_loss.item()))
            fw.write(
                'Train source1 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}\tl1_Loss: {:.6f}\tmmd_loss_m: {:.6f}\tmix_Loss: {:.6f}'.format(
                    i, 100. * i / iteration, loss.item(), cls_loss.item(), mmd_loss.item(), l1_loss.item(),
                    mmd_loss_m.item(),mix_loss.item()))  # 把print内容摘抄下来
            fw.write("\n")
        try:
            source_data, source_label = source2_iter.next()
        except Exception as err:
            source2_iter = iter(source2_loader)
            source_data, source_label = source2_iter.next()
        try:
            target_data, __ = target_iter.next()
        except Exception as err:
            target_iter = iter(target_train_loader)
            target_data, __ = target_iter.next()
        if cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
            target_data = target_data.cuda()
        source_data, source_label = Variable(source_data), Variable(source_label)
        target_data = Variable(target_data)
        optimizer.zero_grad()

        cls_loss, mmd_loss, mmd_loss_m, l1_loss,mix_loss,advloss = model(source_data, target_data, source_label, mark=2,ratio=0.1)
        gamma = 2 / (1 + math.exp(-10 * (i) / (iteration))) - 1
        loss = cls_loss +gamma * (mmd_loss + l1_loss) + mix_ratio * mmd_loss_m+m_ratio_1*mix_loss
        loss=loss+transfer_loss_weight *advloss
        loss.backward()
        optimizer.step()

        if i % log_interval == 0:
            print(
                'Train source2 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}\tl1_Loss: {:.6f}\tmmd_loss_m: {:.6f}\tmix_loss: {:.6f}'.format(
                    i, 100. * i / iteration, loss.item(), cls_loss.item(), mmd_loss.item(), l1_loss.item(),
                    mmd_loss_m.item(),mix_loss.item()))
            fw.write(
                'Train source2 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}\tl1_Loss: {:.6f}\tmmd_loss_m: {:.6f}\tmix_loss: {:.6f}'.format
                (i, 100. * i / iteration, loss.item(), cls_loss.item(), mmd_loss.item(),
                 l1_loss.item(), mmd_loss_m.item(),mix_loss.item()))  # 把print内容摘抄下来
            fw.write("\n")
        if not os.path.exists('state_dict'):
            os.mkdir('state_dict')
        if i % (log_interval * 20) == 0:
            batch= i /(log_interval * 20)
            t_correct = test(model)
            avg = t_correct / 600
            acc_correct += avg
            avg_correct = acc_correct/batch
            if t_correct > correct:
                correct = t_correct
                path = 'state_dict/{0}_{1}_val_acc_{2}'.format(model_name, target_name,correct)
                torch.save(model.state_dict(), path)
            print(source1_name, source2_name, "to", target_name, "%s max correct:" % target_name, correct.item(), "\n")
            fw.write('{}&{} to {} max correct:({:.0f}%)\n'.format(source1_name, source2_name, target_name,
                                                                  correct.item()))  # 把print内容摘抄下来
            print(source1_name, source2_name, "to", target_name, "%s AVG correct:" % target_name, avg_correct.item(), "\n")
            fw.write('{}&{} to {} AVG correct:({:.0f}%)\n'.format(source1_name, source2_name, target_name,
                                                                  avg_correct.item()))


def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    correct1 = 0
    correct2 = 0
    with torch.no_grad():
        for data, target in target_test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            pred1, pred2 = model(data, mark=0)

            pred1 = torch.nn.functional.softmax(pred1, dim=1)
            pred2 = torch.nn.functional.softmax(pred2, dim=1)

            pred = (pred1 + pred2) / 2
            test_loss += F.nll_loss(F.log_softmax(pred, dim=1), target).item()

            pred = pred.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            pred = pred1.data.max(1)[1]
            correct1 += pred.eq(target.data.view_as(pred)).cpu().sum()
            pred = pred2.data.max(1)[1]
            correct2 += pred.eq(target.data.view_as(pred)).cpu().sum()

        len_t = len(target_test_loader.dataset)
        test_loss /= len_t
        print(target_name, '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len_t, 100. * correct / len_t))
        fw.write(target_name)
        fw.write('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len_t, 100. * correct / len_t))
        print('source1 accnum {}, source2 accnum {}'.format(correct1, correct2),
              '\nsource1 Accuracy ({:.0f}%), source2 Accuracy ({:.0f}%)\n'.format(100. * correct1 / len_t,
                                                                                  100. * correct2 / len_t))
        fw.write('source1 accnum {}, source2 accnum {}'.format(correct1, correct2))
        fw.write('\nsource1 Accuracy ({:.0f}%), source2 Accuracy ({:.0f}%)\n'.format(100. * correct1 / len_t,
                                                                                     100. * correct2 / len_t))
    return correct


if __name__ == '__main__':
    model = models.MFSAN(num_classes=2)
    print(model)
    if cuda:
        model.cuda()
    train(model)
    fw.close()