import argparse
import os
# from model_DAVAEdata_01 import Model
# from model_01 import Model
from model_01 import Model

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='wikipedia') # nus-wide / pascal01 / xmedianet/ wikipedia
parser.add_argument("--dual", type=float, default=1.0)
parser.add_argument("--oi", type=float, default=0.9)

parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--gpu', dest='gpu', type=str, default='3', choices=['0', '1', '2', '3'])

parser.add_argument("--num_heads", type=int, default=4, help="number of hidden attention heads") #4,8
parser.add_argument("--num_out_heads", type=int, default=1, help="number of output attention heads")
parser.add_argument("--num_layers", type=int, default=1, help="number of hidden layers") #2
parser.add_argument("--num_hidden", type=int, default=512, help="number of hidden units")#256
parser.add_argument("--residual", action="store_true", default=False, help="use residual connection")
parser.add_argument("--in_drop", type=float, default=.2, help="input feature dropout")
parser.add_argument("--attn_drop", type=float, default=.1, help="attention dropout")
parser.add_argument("--norm", type=str, default=None)
parser.add_argument("--encoder", type=str, default="gat")
parser.add_argument("--decoder", type=str, default="gat")
parser.add_argument("--negative_slope", type=float, default=0.2, help="the negative slope of leaky relu for GAT")# only used in gat, no in dotgat
parser.add_argument("--activation", type=str, default="prelu")

parser.add_argument("--param_loss_recon", type=float, default=1)#10 alpha

parser.add_argument("--param_loss_cls_img", type=float, default=1)#0.01 labmda
parser.add_argument("--param_loss_cls_txt", type=float, default=1)

parser.add_argument("--param_loss_cross", type=float, default=10)#10 beta

parser.add_argument("--param_loss_sim", type=float, default=0)#0.1
parser.add_argument("--param_loss_sim_img", type=float, default=1) # gamma
parser.add_argument("--param_loss_sim_txt", type=float, default=1)#0.1
parser.add_argument("--param_loss_sim_it", type=float, default=0.1) #wikipedia=0.1(complete), pascal=0.1 (ablation=1) (incomplete=1) eta

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

'''
import random
import os
import torch
import numpy as np
# Locking random seed
def seed_setting(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
#seed_setting()
'''
import numpy as np
np.random.seed(57)
import random as rn
rn.seed(57)
import os
os.environ['PYTHONHASHSEED'] = str(57)
import torch
torch.manual_seed(57)
torch.cuda.manual_seed(57)
torch.cuda.manual_seed_all(57)

model = Model(args)
model.train()
model.test()