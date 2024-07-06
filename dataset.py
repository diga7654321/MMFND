import torch
import torch.utils.data as data
import pandas
from tqdm import tqdm
from transformers import BertTokenizer
from pre_train_models.CLIP_BERT import clip
import os
# from img_preprocess import *
from PIL import Image
import cv2
from models import *
# Check if CUDA is available and set the device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"

# print(device)
# Load the CLIP model and preprocessing function
def pre_clip():
    clipmodel, preprocess = clip.load('ViT-B/32', device)
    print('加载clip')
    # Freeze the parameters of the CLIP model
    for param in clipmodel.parameters():
        param.requires_grad = False
    return clipmodel,preprocess

# model_vinvl,dataset_allmap,model_blip,blip_processor = init_models()
# Function to read an image
def read_img(imgs, root_path, LABLEF):
    # GT_path是从imgs里面随机抽的一个名字
    # GT_path = imgs[np.random.randint(0, len(imgs))]
    for img in imgs:
        if '/' in img:
            GT_path = img[img.rfind('/')+1:]
            GT_path = "{}/{}/{}".format(root_path, LABLEF, GT_path)
        if os.path.exists(GT_path):
            # img_GT = Image.open(GT_path).convert('RGB')
            # img_cv = cv2.imread(GT_path)
            return GT_path
        else:
            continue
    return 'data/datasets/weibo/rumor_images/005j4qchjw1eoqppiotwhj30hs0vkgpt.jpg'
    # return Image.open('data/datasets/weibo/rumor_images/005j4qchjw1eoqppiotwhj30hs0vkgpt.jpg').convert('RGB'),cv2.imread('data/datasets/weibo/rumor_images/005j4qchjw1eoqppiotwhj30hs0vkgpt.jpg')

class weibo_dataset(data.Dataset):

    def __init__(self, root_path='./data/datasets/weibo', image_size=224, is_train=True):
        super(weibo_dataset, self).__init__()
        clipmodel, preprocess = load_chinese_clip_model(device)
        self.is_train = is_train
        self.root_path = root_path
        self.index = 0
        self.label_dict = []
        self.preprocess = preprocess
        self.image_size = image_size
        self.local_path = 'data/datasets/weibo'

        # Read data from CSV file
        wb = pandas.read_csv(self.local_path+'/{}_weibo.csv'.format('train' if is_train else 'test'))

        # Store relevant information in label_dict
        # 将相关信息存储在label_dict中
        for i in tqdm(range(len(wb))):
            # 取所有images_name
            images_name = str(wb.iloc[i, 2]).lower()
            label = int(wb.iloc[i, 3])
            content = str(wb.iloc[i, 1])
            sum_content = str(wb.iloc[i, 4])
            record = {}
            record['images'] = images_name
            record['label'] = label
            record['content'] = content
            record['sum_content'] = sum_content

            self.label_dict.append(record)

        assert len(self.label_dict) != 0, 'Error: GT path is empty.'

    def __getitem__(self, index):
        record = self.label_dict[index]
        images, label, content, sum_content = record['images'], record['label'], record['content'], record['sum_content']

        # Determine the label folder
        if label == 0:
            LABLE_F = 'rumor_images'
        else:
            LABLE_F = 'nonrumor_images'
        # print(images)
        imgs = images.split('|')
        try:
            GT_path = read_img(imgs, self.root_path, LABLE_F)
        except Exception:
            raise IOError("Load {} Error {}".format(imgs, record['images']))

        image_clip = self.preprocess(Image.open(GT_path)).unsqueeze(0)
        # img2txt = get_OCR(image_cv) + get_objects(image_cv,self.model_vinvl,self.dataset_allmap) + get_caption(self.model_blip,self.blip_processor,img_GT)
        return (content, image_clip, sum_content), label

    def __len__(self):
        return len(self.label_dict)


# Custom collate function
# 自定义整理功能
def collate_fn_weibo(data):
    sents = [i[0][0] for i in data]
    images = [i[0][1] for i in data]
    # textclip = [i[0][2] for i in data]
    labels = [i[1] for i in data]

    # image = torch.stack(images)
    imageclip = torch.squeeze(torch.stack(images))
    image = imageclip
    labels = torch.LongTensor(labels)

    return sents, image, imageclip, labels
