from __future__ import print_function
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os
import math
import data_loader
import res_mix_adv as models
from confusion_matrix import ConfusionMatrix
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

title=['(a) LC-B (81.33%)',
            '(b) CL-B (80.17%)',
            '(c) CB-L (98.17%)',
            '(d) BC-L (98.33%)',
            '(e) LB-C (99.33%)',
            '(f) BL-C (98.17%)']
model_name=['lcb','clb','cbl','bcl','lbc','blc']
target_name = ["breast","breast","lung","lung","colon","colon"]
path = ['state_dict/lcb_breast_val_acc_488',
        'state_dict/clb_555_breast_val_acc_481',
        'state_dict/cbl_22_lung_val_acc_589',
        'state_dict/bcl_lung_val_acc_590',
        'state_dict/lbc_22_colon_val_acc_596',
        'state_dict/blc_22_colon_val_acc_589']

# Training settings
batch_size = 32
cuda = True
seed = 8

root_path = "../dataset_300/"
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
target_test_loader = data_loader.load_testing(root_path, target_name[5], batch_size, kwargs)
def test(model,i):
    model.eval()
    test_loss = 0
    correct = 0
    correct1 = 0
    correct2 = 0
    matrix = ConfusionMatrix(2, [0, 1])
    with torch.no_grad():
        for data, target in target_test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            pred1, pred2 = model(data, mark=0)
            pred1 = torch.nn.functional.softmax(pred1, dim=1)
            pred2 = torch.nn.functional.softmax(pred2, dim=1)
            pred = (pred1 + pred2) / 2
            pred_s = F.log_softmax(pred, dim=1)
            test_loss += F.nll_loss(pred_s, target).item()
            pred = pred.data.max(1)[1]
            matrix.update(pred, target)
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            pred = pred1.data.max(1)[1]
            correct1 += pred.eq(target.data.view_as(pred)).cpu().sum()
            pred = pred2.data.max(1)[1]
            correct2 += pred.eq(target.data.view_as(pred)).cpu().sum()
        matrix.summary()
        #matrix.plot(model_name[i],title[i])
        len_t = len(target_test_loader.dataset)
        test_loss /= len_t
        print(target_name[i], '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len_t, 100. * correct / len_t))

        print('source1 accnum {}, source2 accnum {}'.format(correct1, correct2),
              '\nsource1 Accuracy ({:.0f}%), source2 Accuracy ({:.0f}%)\n'.format(100. * correct1 / len_t,
                                                                                  100. * correct2 / len_t))
    return correct
if __name__ == '__main__':
    model = models.MFSAN(num_classes=2)
    if cuda:
        model.cuda()
    model.load_state_dict(torch.load(path[5]))
    t_correct = test(model,5)