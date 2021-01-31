import numpy as np
import torch

from flask import Flask, request
from flask_cors import CORS
from flask import jsonify

from mmcv import Config
from mmcv.runner import load_checkpoint

from mmfashion.core import AttrPredictor, CatePredictor
from mmfashion.models import build_predictor
from mmfashion.utils import get_img_tensor

app = Flask(__name__)
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


@app.route('/annotate', methods=(['POST']))
def annotate():
    file = request.files.get('image')
    img_tensor = get_img_tensor(file, False)

    # predict probabilities for each attribute
    fine_attr_prob, cate_prob = model_fine(
        img_tensor, attr=None, landmark=landmark_tensor, return_loss=False)

    fine_attrs = jsonResult(fine_attr_prob, fine_attr_predictor.attr_idx2name)
    cats = jsonResult(cate_prob, cate_predictor.cate_idx2name)

    coarse_attr_prob = model_coarse(
        img_tensor, attr=None, landmark=landmark_tensor, return_loss=False)

    coarse_attr_predictor.show_prediction(coarse_attr_prob)

    resultDict = {}
    resultDict['attributes'] = fine_attrs
    resultDict['categories'] = cats

    return jsonify(resultDict)


if __name__ == '__main__':
    cfg_fine = Config.fromfile(
        './configs/category_attribute_predict/global_predictor_vgg.py')
    cfg_coarse = Config.fromfile(
        './configs/attribute_predict_coarse/roi_predictor_resnet_attr.py')

    # global attribute predictor will not use landmarks
    # just set a default value
    landmark_tensor = torch.zeros(8)

    model_fine = build_predictor(cfg_fine.model)
    load_checkpoint(model_fine, './checkpoint/vgg16_fine_global.pth',
                    map_location='cpu')

    model_coarse = build_predictor(cfg_coarse.model)
    load_checkpoint(model_coarse, './checkpoint/coarse-resnet-landmark.pth',
                    map_location='cpu')

    print('Models loaded.')

    model_fine.eval()
    model_coarse.eval()

    fine_attr_predictor = AttrPredictor(cfg_fine.data.test)
    cate_predictor = CatePredictor(cfg_fine.data.test)

    coarse_attr_predictor = AttrPredictor(cfg_coarse.data.test)

    app.run()
