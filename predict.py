# 该模块主要是为了预测推理的，输入一个图像得到一个浅层或者输入浅层得到一个图像
'''
# Part1 引入相关的模型
'''
import torch
from dataset import Mnist_dataset
import matplotlib.pyplot as plt

'''
# part2 下载模型
'''
net = torch.load('AE_encoder_eopch_20.pt')
net.eval()
net1 = torch.load('AE_decoder_eopch_20.pt')
net1.eval()
data_cs = Mnist_dataset

'''
# Part3 开始测试
'''
if __name__ == '__main__':
    with torch.no_grad():
        img, label = data_cs[2]
        # 开始绘制初始的图像
        plt.imshow(img.permute(2, 1, 0))
        plt.show()

        img=img.unsqueeze(0)
        latent_predict = net(img.view(1,-1))
        print(latent_predict.size())
        # 生成中间1*10的浅层，生成对应的图像
        latent_cs=latent_predict
        img_predict=net1(latent_cs)
        result = img_predict.view(img.size()[1],img.size()[2],img.size()[3])
        print(result.size())

        # 开始绘制结果图像
        plt.imshow(result.permute(2,1,0))
        plt.show()

