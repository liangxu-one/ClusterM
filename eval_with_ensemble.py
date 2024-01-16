import os
import torch
from torch.utils.data import DataLoader
from config import Config
from model import CaptionModel
from read_file import build_data
from evaluation import compute_scores
from module.decoder_with_ensemble import greedy_decode, beam_decode

def printAns(gts, res, imgid_to_filename):
    for key in res.keys():
        print("image_id: ", key)
        print("file_name: ", imgid_to_filename[key])
        print("res: ", res[key])
        print("gts: ", gts[key])

def eval(config, model = None, val_data = None, val_dict = None):

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
    val_dict = build_data(config)
    val_data = DataLoader(val_dict, config.batch_size, shuffle = False, num_workers = config.num_workers)
    print("val data is: ", len(val_dict))
    print("读取数据结束")

    gts = {}
    res = {}
    imgid_to_filename = {}

    decode = greedy_decode if config.decode_method == 'greedy' else beam_decode

    for i, batch in enumerate(val_data):

        img = batch[0].to(device)
        caption_index = batch[1].to(device)
        caption_index = decode(model_list, img, caption_index, config)

        pred_str = config.tokenizer.batch_decode(caption_index.tolist(), skip_special_tokens=True)

        bs = img.size(0)
        for k in range(bs):
            image_id = int(batch[2][k])
            gts[image_id] = val_dict.imgid_to_sentences[image_id]
            res[image_id] = [pred_str[k]]
            imgid_to_filename[image_id] = batch[3][k]

    # printAns(gts, res, imgid_to_filename)
    score = compute_scores(gts, res)
    print(score[0])

if __name__ == '__main__':
    config = Config(TrainOrVal = 'test')
    with torch.no_grad():
        eval(config)