import numpy as np
import torch

from flask import Flask, request, send_from_directory
from flask_cors import CORS
from flask import jsonify

from mmcv import Config
from mmcv.runner import load_checkpoint

from mmfashion.core import AttrPredictor, CatePredictor, ClothesRetriever
from mmfashion.models import build_predictor, build_retriever
from mmfashion.datasets import build_dataloader, build_dataset
from mmfashion.utils import get_img_tensor

import colorgram, webcolors

app = Flask(__name__, static_url_path='/data')
CORS(app)


@app.route('/')
def hello():
    return "Hello World!"


def jsonResult(pred, idx2name):
    data = pred.data.cpu().numpy().reshape((-1))
    results = [(x, i) for i, x in enumerate(data)]
    # results.sort()

    myDict = {}
    for result in results:
        myDict[idx2name[result[1]]] = float(result[0])

    return myDict

def closest_color(requested_color):
    min_colors = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

def get_color_name(requested_color):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_color)
    except ValueError:
        closest_name = closest_color(requested_color)
        actual_name = None
    return actual_name, closest_name

def _process_embeds(dataset, model, cfg, use_cuda=True):
    data_loader = build_dataloader(
        dataset,
        cfg.data.imgs_per_gpu,
        cfg.data.workers_per_gpu,
        len(cfg.gpus.test),
        dist=False,
        shuffle=False)

    print(cfg.data)
    embeds = []
    with torch.no_grad():
        i = 0
        print("Data loader size: " + str(len(data_loader)))
        for data in data_loader:
            print(i)
            i += 1
            img = data['img']
            if use_cuda:
                img = data['img'].cuda()
            embed = model(img, landmark=data['landmark'], return_loss=False)
            embeds.append(embed)

    embeds = torch.cat(embeds)
    embeds = embeds.data.cpu().numpy()
    return embeds


@app.route('/img/<path:path>')
def send_img(path):
    return send_from_directory('data', path)


@app.route('/retrieve', methods=(['POST']))
def retrieve():
    file = request.files.get('image')
    img_tensor = get_img_tensor(file, True)
 
    query_feat = model_ret(img_tensor, landmark=None, return_loss=False)
    query_feat = query_feat.data.cpu().numpy()
    gallery_set = build_dataset(cfg_ret.data.gallery)
    gallery_embeds = _process_embeds(gallery_set, model_ret, cfg_ret)
    retriever = ClothesRetriever(cfg_ret.data.gallery.img_file, cfg_ret.data_root,
                                 cfg_ret.data.gallery.img_path)

    result = retriever.show_retrieved_images(query_feat, gallery_embeds)
    resultDict = {}
    resultDict['paths'] = result
    return jsonify(resultDict)

@app.route('/annotate', methods=(['POST']))
def annotate():
    file = request.files.get('image')
    img_tensor = get_img_tensor(file, False)

    # predict probabilities for each attribute
    fine_attr_prob, cate_prob = model_fine(
        img_tensor, attr=None, landmark=landmark_tensor, return_loss=False)

    coarse_attr_prob = model_coarse(
        img_tensor, attr=None, landmark=landmark_tensor, return_loss=False)

    coarse_attrs = jsonResult(
        coarse_attr_prob, coarse_attr_predictor.attr_idx2name)
    cats = jsonResult(cate_prob, cate_predictor.cate_idx2name)

    colors = colorgram.extract(file, 5)
    color_names = [get_color_name(x.rgb)[1] for x in colors]

    resultDict = {}
    resultDict['attributes'] = coarse_attrs
    resultDict['categories'] = cats
    resultDict['colors'] = color_names

    return jsonify(resultDict)


if __name__ == '__main__':
    cfg_fine = Config.fromfile(
        './configs/category_attribute_predict/global_predictor_vgg.py')
    cfg_coarse = Config.fromfile(
        './configs/attribute_predict_coarse/global_predictor_resnet_attr.py')
    cfg_ret = Config.fromfile('configs/retriever_in_shop/global_retriever_vgg_loss_id.py')
    
    # global attribute predictor will not use landmarks
    # just set a default value
    landmark_tensor = torch.zeros(8)

    model_fine = build_predictor(cfg_fine.model)
    load_checkpoint(model_fine, './checkpoint/vgg16_fine_global.pth',
                    map_location='cpu')

    model_coarse = build_predictor(cfg_coarse.model)
    load_checkpoint(model_coarse, './checkpoint/resnet_coarse_global.pth',
                    map_location='cpu')


    model_ret = build_retriever(cfg_ret.model).cuda()
    load_checkpoint(model_ret, 'checkpoint/Retrieve/vgg/global/epoch_100.pth', map_location=torch.device('cuda:0'))

    print('Models loaded.')

    model_fine.eval()
    model_coarse.eval()
    model_ret.eval()

    cate_predictor = CatePredictor(cfg_fine.data.test)
    coarse_attr_predictor = AttrPredictor(cfg_coarse.data.test)

    app.run(host="0.0.0.0", port=80)
