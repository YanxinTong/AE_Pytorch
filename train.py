# 训练模型，该模块主要是为了实现对于模型的训练，
'''
# Part1 引入相关的库函数
'''

import torch
from torch import nn
from dataset import Mnist_dataset
from AE import AE
import torch.utils.data as data

'''
初始化一些训练参数
'''
EPOCH = 50
Mnist_dataloader = data.DataLoader(dataset=Mnist_dataset, batch_size=64, shuffle=True)

# 前向传播的模型
net = AE(img_channel=1, img_size=28,encode_f1_size=400,latent_size=10)

# 计算损失函数
loss = nn.L1Loss()

# 反向更新参数
lr = 1e-3
optim = torch.optim.Adam(params=net.parameters(), lr=lr)

'''
# 开始训练
'''
# net.train() # 设置为训练模式

for epoch in range(EPOCH):
    n_iter = 0
    for batch_img,_ in Mnist_dataloader:
        # 先进行前向传播
        batch_img_pre=net(batch_img) #

        # 计算损失
        loss_cal=loss(batch_img_pre,batch_img)

        # 清除梯度
        optim.zero_grad()
        # 反向传播
        loss_cal.backward()
        # 更新参数
        optim.step()

        l=loss_cal.item()


        if n_iter%100==0:
            print('此时的epoch为{},iter为{},loss为{}'.format(epoch,n_iter,l))

        n_iter += 1
    if epoch==20:
        # 注意pt文件是保存整个模型及其参数的，pth文件只是保存参数
        torch.save(net.encode,'AE_encoder_eopch_{}.pt'.format(epoch))
        # 注意pt文件是保存整个模型及其参数的，pth文件只是保存参数
        torch.save(net.decode, 'AE_decoder_eopch_{}.pt'.format(epoch))