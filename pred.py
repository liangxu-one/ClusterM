import os
import torch
from PIL import Image
from config import Config
from model import CaptionModel
from module.decoder import greedy_decode, beam_decode

def pred(config, image_path):

    device = torch.device("cuda:{}".format(config.gpu) if config.use_cuda else "cpu")

    model = CaptionModel(config)
    model.load_state_dict(torch.load(os.path.join(config.model_save_path, config.ck), map_location='cpu'))
    model.to(device)

    model.eval()

    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert("RGB")
    img = config.image_process(image, return_tensors = 'pt')['pixel_values'].to(device)
    caption_index = torch.tensor([config.bos_token_id]).unsqueeze(0).to(device)

    decode = greedy_decode if config.decode_method == 'greedy' else beam_decode
    caption_index = decode(model, img, caption_index, config)

    pred_str = config.tokenizer.batch_decode(caption_index.tolist(), skip_special_tokens=True)

    print(image_path, pred_str)

if __name__ == '__main__':
    config = Config(TrainOrVal = 'test')
    image_path = '/home2/lx/image_caption_ablation/test/COCO_test2014_000000000001.jpg'
    with torch.no_grad():
        pred(config, image_path)