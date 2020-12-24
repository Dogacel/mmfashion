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
    attr_prob, cate_prob = model(
        img_tensor, attr=None, landmark=landmark_tensor, return_loss=False)

    attrs = jsonResult(attr_prob, attr_predictor.attr_idx2name)
    cats = jsonResult(cate_prob, cate_predictor.cate_idx2name)

    resultDict = {}
    resultDict['attributes'] = attrs
    resultDict['categories'] = cats

    return jsonify(resultDict)


if __name__ == '__main__':
    cfg = Config.fromfile(
        './configs/category_attribute_predict/global_predictor_vgg.py')

    # global attribute predictor will not use landmarks
    # just set a default value
    landmark_tensor = torch.zeros(8)

    model = build_predictor(cfg.model)
    load_checkpoint(model, './checkpoint/vgg16_fine_global.pth',
                    map_location='cpu')
    print('Model loaded.')

    model.eval()

    attr_predictor = AttrPredictor(cfg.data.test)
    cate_predictor = CatePredictor(cfg.data.test)

    app.run()
