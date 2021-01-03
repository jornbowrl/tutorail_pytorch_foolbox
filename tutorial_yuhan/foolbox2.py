#!/usr/bin/env python
# coding: utf-8

# In[7]:


import torchvision.models as models
import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD
import foolbox as fb
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
from PIL import Image
import os 
import numpy as np 
import tqdm 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[12]:
class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


acc_torch = lambda x,y: torch.eq(torch.argmax(x,dim=-1),y)

def predict_func (dataloader , model ):
    with torch.no_grad():
        acc_list=  []
        for img,lbl in dataloader:
            img = img.to(device)
            lbl = lbl.to(device)
            
            pre= model.forward(img)
            
            v=acc_torch(pre,lbl)
            acc_list.append(v.cpu().numpy())
    acc_list=np.concatenate(acc_list)
    
    clean_acc=float(np.mean(acc_list))
    print(f" accuracy (pytorch):  {clean_acc * 100:.1f} %")

    return np.mean(acc_list)
            

def main() -> None:
    # instantiate a model (could also be a TensorFlow or JAX model)
    #model = models.resnet18(pretrained=True).eval()
    #model=torch.load('/data1/zyh/copycat/Framework/cifar_model.pth')

    model =AlexNet()
    path = "./cifar_net.pth"
    #path = '/data1/zyh/copycat/Framework/cifar_model.pth'
    #model.load_state_dict(torch.load('/data1/zyh/copycat/Framework/cifar_model.pth'))
    #pretrained_dict = {k: v for k, v in model_pretrained.items() if k in model_dict}
    #model_dict.update(pretrained_dict)
    #model.load_state_dict(state_dict)
    model.load_state_dict(torch.load(path),strict=True)
    model=model.to(device)
    model.eval()

    print(type(model))
    #preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    preprocessing = dict(mean=[0.5]*3, std=[0.5]*3, axis=-3)
    fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)


    # get data and test the model
    # wrapping the tensors with ep.astensors is optional, but it allows
    # us to work with EagerPy tensors in the following
    test_dataset = torchvision.datasets.CIFAR10(root='~/.torch/',
                                             train=False,
                                             #transform = transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()]),
                                             transform = transforms.Compose([transforms.ToTensor()]),
                                             download=True)
#     test_dataset .data = test_dataset.data[:128*5]
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=128, #该参数表示每次读取的批样本个数
                                           shuffle=False) #该参数表示读取时是否打乱样本顺序
                                           # 创建迭代器
    #data_iter = iter(test_loader)

    #images, labels = next(data_iter)
    # 当迭代开始时, 队列和线程开始读取数据
    #images, labels = data_iter.next()
    #im=images
    #images=im.resize(100,3,128,128)
    with torch.no_grad():
        all_clean_acc_foolbox= []

        ## native predict 
        predict_func(test_loader, model)
        
        for ii,(imgs, lbls) in tqdm.tqdm(enumerate(test_loader),total=len(test_loader)):
            imgs=imgs.to(device)
            lbls=lbls.to(device)
            
            images, labels = ep.astensors(imgs,lbls)
            
            ##  calc with foolbox  
            pred_lbl_foolbox = fmodel(images)
            clean_acc_one = accuracy(fmodel, imgs, lbls)
            all_clean_acc_foolbox.append(clean_acc_one)
            
        clean_acc= sum(all_clean_acc_foolbox)/len(all_clean_acc_foolbox)
        
    print(f"clean accuracy:  {clean_acc * 100:.1f} %")

    # apply the attack
    attack = LinfPGD()
    '''epsilons = [
        0.0,
        0.0002,
        0.0005,
        0.0008,
        0.001,
        0.0015,
        0.002,
        0.003,
        0.01,
        0.1,
        0.3,
        0.5,
        1.0,
    ]'''
    epsilons = [
        0.0005,
        0.001,
        0.002,
        0.01,
        0.1,
    ]
    def attack_one_batch(fmodel,images, labels,iter=0,verbose=True):
        images, labels = ep.astensors(images,labels)

        raw_advs, clipped_advs, success = attack(fmodel, images, labels, epsilons=epsilons)
        if verbose: print ("==="*8,iter,"==="*8)
        if verbose:
            robust_accuracy = 1 - success.float32().mean(axis=-1)
            print("robust accuracy for perturbations with")
            for eps, acc in zip(epsilons, robust_accuracy):
                print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")
    
        if verbose:
            fig = plt.gcf()
            os.makedirs("./image/",exist_ok=True)
            for i in range(len(raw_advs)):
                img_v = raw_advs[i].raw
                torchvision.utils.save_image(img_v, f'./image/{str(iter).zfill(4)}_{str(i).zfill(3)}_.png')
        return [x.raw for x in raw_advs] #
    
    print ("===="*8,"start attack","===="*8)
    collection_adv= []
    collection_gt= []
    for ii,(imgs, lbls) in tqdm.tqdm(enumerate(test_loader),total=len(test_loader)):
        imgs=imgs.to(device)
        lbls=lbls.to(device)
        
#         images, labels = ep.astensors(images,labels)
        adv_ret = attack_one_batch(fmodel=fmodel,
                         images=imgs,
                          labels=lbls,iter=ii,verbose=True)

        collection_adv.append(torch.stack(adv_ret) )
        collection_gt.append(lbls.cpu())
    
    print ("===="*8,"start evaluation","===="*8)
    with torch.no_grad():
        
        adv_total_dataset = torch.cat(collection_adv,dim=1)
        lbl_total_dataset = torch.cat(collection_gt).to(device)
        
#         print (adv_total_dataset.mean(dim=(1,2,3,4)),"the mean if each eps")
        for (eps,ep_adv_dataset) in  zip(epsilons,adv_total_dataset):
#             print ("eps:",eps,"===>"*8)
#             print (ep_adv_dataset.mean(),"each...")
            advs_= ep_adv_dataset.to(device)
            acc2 = accuracy(fmodel, advs_, lbl_total_dataset)
            print(f"  Linf norm ≤ {eps:<6}: {acc2 * 100:4.1f} %")
            dataset= torch.utils.data.TensorDataset(ep_adv_dataset,lbl_total_dataset)
            dl = torch.utils.data.DataLoader(dataset,batch_size=128)
            predict_func(dl,model)


if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:





# In[ ]:




