import os
from core.predict import Predictor
from core.train import get_random_images


def test_bla():
    weights_path = os.path.join(os.getcwd(), "model", "stage3_256px_1800images.h5")
    # weights_path = os.path.join(os.getcwd(), "model", "stage3_256px_overfitted.h5")
    assert os.path.isfile(weights_path)
    p = Predictor(weights_path)
    # images_test, images_validation = get_random_images(limit=10)
    # p.predict()
    img_path = os.path.join(os.getcwd(), "test", "data", "18_139423_171197.tiff")
    res = p.predict_path(img_path)
    print("# Rois: ", len(res[0]["rois"]))
    print("# Masks: ", len(res[0]["masks"]))
    print("# Scores: ", len(res[0]["scores"]))
    assert 1 == 1
