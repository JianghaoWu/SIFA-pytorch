#Evaluate of SIFA
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
from model import SIFA
import yaml
from utils import SingleDataset
from metrics import dice_eval,assd_eval,create_visual_anno
import cv2
from utils import parse_config
import os

config = "/data2/jianghao/Two/SIFA/config/train.cfg"
config = parse_config(config)
exp_name = config['train']['exp_name']

def norm_01(image):
    mn = np.min(image)
    mx = np.max(image)
    image = (image-mn)/(mx-mn).astype(np.float32)
    return image
    
def save_img(image):
    image = norm_01(image)
    image = (image*255).astype(np.uint8)
    return image    

    
device = torch.device('cuda:{}'.format(config['test']['gpu']))
test_path = config['test']['test_path']
num_classes = config['test']['num_classes']
sifa_model = SIFA(config).to(device)
sifa_model.load_state_dict(torch.load('{}'.format(config['test']['test_model'])))
sifa_model.eval()
#test dataset
test_dataset = SingleDataset(test_path)
batch_size = config['test']['batch_size']
test_loader = DataLoader(test_dataset,batch_size,shuffle=False)

#test
all_batch_dice = []
all_batch_assd = []
with torch.no_grad():
    for it,(xt,xt_label) in enumerate(test_loader):        
        xt = xt.to(device)
        xt_label = xt_label.numpy().squeeze().astype(np.uint8)
        output = sifa_model.test_seg(xt).detach()
        output = output.squeeze(0)
        output = torch.argmax(output,dim=0)        
        output = output.cpu().numpy()
        

        xt = xt.detach().cpu().numpy().squeeze()
        gt = xt_label.reshape(256,256).astype(np.uint8)
        output = output.squeeze()
        xt = save_img(xt)
        
        output_vis = create_visual_anno(output)
        gt_vis = create_visual_anno(gt)
        results = "results/" + str(exp_name)
        if(not os.path.exists(results)):
                os.mkdir(results)
        cv2.imwrite('{}/xt-{}.jpg'.format(results, it+1),xt)
        cv2.imwrite('{}/gt-{}.jpg'.format(results, it+1),gt_vis)
        cv2.imwrite('{}/output-{}.jpg'.format(results, it+1),output_vis)
        

        
        one_case_dice = dice_eval(output,xt_label,num_classes) * 100
        #print('{:.4f} th case dice MYO:{:.4f} LV:{:.4f} RV:{:.4f}'.format(it+1,one_case_dice[0],one_case_dice[1],one_case_dice[2]))
        #dicefile.write('file:{},{} th case dice:{}\n'.format(filename,it+1,one_case_dice))
        all_batch_dice += [one_case_dice]
        try:
            one_case_assd = assd_eval(output,xt_label,num_classes)
        except:
            continue
        all_batch_assd.append(one_case_assd)
        
        

all_batch_dice = np.array(all_batch_dice)
all_batch_assd = np.array(all_batch_assd)
mean_dice = np.mean(all_batch_dice,axis=0) 
std_dice = np.std(all_batch_dice,axis=0) 
mean_assd = np.mean(all_batch_assd,axis=0)
print(all_batch_assd)
std_assd = np.std(all_batch_assd,axis=0)
print('-----------')
print('MYO||LV||RV')
print('Dice mean:{}'.format(mean_dice))
print('Dice std:{}'.format(std_dice))
print('total mean dice:',np.mean(mean_dice))
print('ASSD mean:{}'.format(mean_assd))
print('ASSD std:{}'.format(std_assd))
print('total mean assd:',np.mean(mean_assd))
print('-----------')
