import os
import sys
from core.mask_rcnn_config import VALIDATION_DATA_DIR, TEST_DATA_DIR
from pycocotools.coco import COCO
from core.cocoeval import COCOeval
import json
from core.predict import test_images


def evaluate(annotation_dir):
    annotation_path = os.path.join(annotation_dir, "annotation.json")
    predictions_path = os.path.join(os.getcwd(), "predictions.json")
    with open(predictions_path, 'r', encoding="utf-8") as f:
        predicitions = json.load(f)
    ids = list(map(lambda p: p["image_id"], predicitions))

    ground_truth_annotations = COCO(annotation_path)

    assert os.path.isfile(annotation_path)
    # with open(predictions_path, 'r', encoding="utf-8") as f:
    #     data = f.read()
    #     submission_file = json.loads(data)
    results = ground_truth_annotations.loadRes(predictions_path)
    cocoEval = COCOeval(ground_truth_annotations, results, 'segm')

    cocoEval.evaluate()
    cocoEval.accumulate()
    average_precision = cocoEval._summarize(ap=1, iouThr=0.5, areaRng="all", maxDets=100)
    average_recall = cocoEval._summarize(ap=0, iouThr=0.5, areaRng="all", maxDets=100)
    print("Average Precision : {} || Average Recall : {}".format(average_precision, average_recall))
    for image_id in ids:
        iou = cocoEval.computeIoU(image_id, 100)
        print("IoU: {}".format(iou))


if __name__ == "__main__":
    # predictions_path = os.path.join(os.getcwd(), "eval_predictions.json")
    # images_path = os.path.join(os.getcwd(), "eval_tested_images.txt")
    # if os.path.isfile(predictions_path):
    #     os.remove(predictions_path)
    # if os.path.isfile(images_path):
    #     os.remove(images_path)

    # test_images("predictions.json", "eval_tested_images.txt", 5, VALIDATION_DATA_DIR)
    annotation_dir = TEST_DATA_DIR
    if len(sys.argv) > 1:
        annotation_dir = sys.argv[1]

    evaluate(annotation_dir)
