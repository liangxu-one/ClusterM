import os, sys
import json
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from config import Config
from model import CaptionModel
from module.decoder_with_ensemble import greedy_decode, beam_decode

path = sys.path[0]

class OnlineDataset(Dataset):
    def __init__(self, config) -> None:
        super(OnlineDataset, self).__init__()

        self.coco_online_dataset = os.path.join(path, 'coco_online_data/image_info_test2014.json')
        self.coco_online_img_dir = os.path.join(path, 'coco_online_data/test2014/')

        file = open(self.coco_online_dataset, encoding='utf-8')
        content = json.load(file)

        self.dataset = [] 
        for item in content['images']:
            temp = {}
            temp['img_id'] = int(item['id'])
            temp['file_name'] = item['file_name']
            self.dataset.append(temp)

        self.transforms = config.image_process

        self.tokenizer = config.tokenizer
        self.pad_token_id = config.pad_token_id
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id

    def __getitem__(self, index):
        item = self.dataset[index]
        image_path = self.coco_online_img_dir + item['file_name']
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert("RGB")
        image = self.transforms(image, return_tensors = 'pt')['pixel_values'].squeeze(0)
        caption_index = torch.tensor([self.bos_token_id])

        return image, caption_index, item['img_id']

    def __len__(self):
        return len(self.dataset)
        # return 10

def eval(config):

    device = torch.device("cuda:{}".format(config.gpu) if config.use_cuda else "cpu")

    model_list = []
    for i in range(len(config.ck_list)):
        model = CaptionModel(config)
        model.load_state_dict(torch.load(os.path.join(config.model_save_path, config.ck_list[i]), map_location='cpu'))
        model.to(device)
        model_list.append(model)

    for i in range(len(model_list)):
        model_list[i].eval()

    # 读取数据
    print("读取数据")
    online_dict = OnlineDataset(config)
    online_data = DataLoader(online_dict, config.batch_size, shuffle = False, num_workers = config.num_workers)
    print("online data is: ", len(online_dict))
    print("读取数据结束")

    res = []

    decode = greedy_decode if config.decode_method == 'greedy' else beam_decode

    for i, batch in enumerate(online_data):

        img = batch[0].to(device)
        caption_index = batch[1].to(device)
        caption_index = decode(model_list, img, caption_index, config)

        pred_str = config.tokenizer.batch_decode(caption_index.tolist(), skip_special_tokens=True)

        bs = img.size(0)
        for k in range(bs):
            temp = {}
            temp['image_id'] = int(batch[2][k])
            temp['caption'] = str(pred_str[k])
            res.append(temp)

        if i % 100 == 0:
            print('i/batch: {}/{}'.format(i, len(online_data)))

    file = open(os.path.join(path, 'coco_online_data/result.json'), mode = 'w')
    file.write(json.dumps(res))
    print("预测结束")

if __name__ == '__main__':
    config = Config(TrainOrVal = 'test')
    with torch.no_grad():
        eval(config)