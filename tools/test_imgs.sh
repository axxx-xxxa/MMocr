#!/bin/bash

# DATE=`date +%Y-%m-%d`
# TIME=`date +"%H-%M-%S"`

# if [ $# -lt 5 ]
# then
#     echo "Usage: bash $0 CONFIG CHECKPOINT IMG_ROOT_PATH IMG_LIST OUT_DIR"
#     exit
# fi

# CONFIG_FILE=$1
# CHECKPOINT=$2
# IMG_ROOT_PATH=$3
# IMG_LIST=$4
# OUT_DIR=$5

# mkdir ${OUT_DIR} -p &&


# python tools/test_imgs.py \
#      ${CONFIG_FILE} ${CHECKPOINT} ${IMG_ROOT_PATH} ${IMG_LIST} \
#       --out-dir ${OUT_DIR}


# python demo/image_demo.py \
#         img_list.txt \
#         configs/textrecog/crnn/crnn_academic_dataset.py \
#         configs/textrecog/crnn/crnn_academic-a723a1c5.pth \
#         ./demo
python demo/image_demo.py \
        img_list.txt \
        configs/textdet/textsnake/textsnake_r50_fpn_unet_1200e_ctw1500.py \
        configs/textdet/textsnake/textsnake_r50_fpn_unet_1200e_ctw1500-27f65b64.pth \
        ./textsnake_dec_result
# python tools/test_imgs.py \
#         configs/textrecog/crnn/crnn_academic_dataset.py \
#         configs/textrecog/crnn/crnn_academic-a723a1c5.pth \
#         qtests \
#         img_list.txt \
#         --out-dir qtests_det
