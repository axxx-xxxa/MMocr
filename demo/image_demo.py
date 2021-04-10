from argparse import ArgumentParser

import mmcv
import time
from mmdet.apis import init_detector
from mmocr.apis.inference import model_inference
from mmocr.datasets import build_dataset  # noqa: F401
from mmocr.models import build_detector  # noqa: F401


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file.')
    parser.add_argument('config', help='Config file.')
    parser.add_argument('checkpoint', help='Checkpoint file.')
    parser.add_argument('save_path', help='Path to save visualized image.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference.')
    parser.add_argument(
        '--imshow',
        action='store_true',
        help='Whether show image with OpenCV.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    if model.cfg.data.test['type'] == 'ConcatDataset':
        model.cfg.data.test.pipeline = model.cfg.data.test['datasets'][
            0].pipeline

    # test a single image
    print(args.img)
    s=time.time()
    with open(args.img, 'r') as ff:
        lines = ff.readlines()
    for i, line in enumerate(lines):
        imgPath = line.strip()
        imgPath = f"qtests/{imgPath}"
        imgName = imgPath.split('/')[-1]
        print(imgPath)
        result = model_inference(model, imgPath)
        print(result)

        #show the results
        img = model.show_result(imgPath, result, out_file=None, show=False)
        print("time",time.time()-s)
        mmcv.imwrite(img, f"{args.save_path}/{imgName}.jpg")

        if args.imshow:
            mmcv.imshow(img, 'predicted results')



if __name__ == '__main__':
    main()
