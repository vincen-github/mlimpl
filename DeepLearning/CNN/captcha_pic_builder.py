#coding:utf8
from captcha.image import ImageCaptcha
from random import randint
import os
import csv

def captcha_pic_builder():
    '''
    生成序列验证码图片及包含其对应目标值的csv文件，哈哈哈....
    :return:
    '''
    list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
            'W', 'X', 'Y', 'Z']
    #因为csv标签名不能放在循环里的缘故，故将创建及增加内容放在外面
    #newline='',是为了防止以Excel打开时会多出空行
    with open('./captcha_pic/labels.csv', 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile, dialect='excel')
        writer.writerow(['file_num', 'chars'])
        for j in range(20):
            chars = ''
            for i in range(4):
                chars += list[randint(0, 25)]
                print(chars)
            #生成图片
            image = ImageCaptcha().generate_image(chars)
            # image.show()
            filename = str(j) +'.jpg'
            #将图片保存到本地
            image.save(os.path.join('./captcha_pic/', filename))
            #添加样本序列及目标值
            writer.writerow([j,chars])
    return None

if __name__ == '__main__':
    captcha_pic_builder()


