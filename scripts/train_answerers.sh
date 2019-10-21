#!/bin/bash
# Usage: `./scripts/train_answerers.sh color`

BASE_DIR=$(pwd)



if [ "$1" == 'all_categories' ]
then
	echo "Training Answerer on all categories"
	python train.py -trainMode sl-abot -useGPU \
					-inputQues ../data/visdial_submodule/data/visdial_data.h5 \
					-inputJson ../data/visdial_submodule/data/visdial_params.json \
					-inputImg ../data/visdial_submodule/data/image_feats_res101.h5 \
					-cocoDir ../data/visdial_submodule/data/visdial_images \
					-cocoInfo ../data/visdial_submodule/data/visdial_images/coco_info.json \
					-categoryMap ../data/visdial_submodule/data/qa_category_mapping.json  \
					-imgNorm 0 \
					-imgFeatureSize 2048 \
					-enableVisdom 1 \
					-visdomServerPort 8895 \
					-visdomEnv all_categories_v3 \
					-saveName all_categories_v3

elif [ "$1" == 'binary' ]
then
	echo "Training Answerer on binary questions"
	python train.py -trainMode sl-abot -useGPU \
					-inputQues ../data/visdial_submodule/data/visdial_data.h5 \
					-inputJson ../data/visdial_submodule/data/visdial_params.json \
					-inputImg ../data/visdial_submodule/data/image_feats_res101.h5 \
					-cocoDir ../data/visdial_submodule/data/visdial_images \
					-cocoInfo ../data/visdial_submodule/data/visdial_images/coco_info.json \
					-categoryMap ../data/visdial_submodule/data/qa_category_mapping.json  \
					-imgNorm 0 \
					-imgFeatureSize 2048 \
					-enableVisdom 1 \
					-visdomServerPort 8895 \
					-visdomEnv binary_v3 \
					-qaCategory binary \
					-saveName binary_v3

elif [ "$1" == 'color' ]
then
	echo "Training Answerer on color questions"
	python train.py -trainMode sl-abot -useGPU \
					-inputQues ../data/visdial_submodule/data/visdial_data.h5 \
					-inputJson ../data/visdial_submodule/data/visdial_params.json \
					-inputImg ../data/visdial_submodule/data/image_feats_res101.h5 \
					-cocoDir ../data/visdial_submodule/data/visdial_images \
					-cocoInfo ../data/visdial_submodule/data/visdial_images/coco_info.json \
					-categoryMap ../data/visdial_submodule/data/qa_category_mapping.json  \
					-imgNorm 0 \
					-imgFeatureSize 2048 \
					-enableVisdom 1 \
					-visdomServerPort 8895 \
					-visdomEnv color_v3 \
					-qaCategory color \
					-saveName color_v3

elif [ "$1" == 'count' ]
then
	echo "Training Answerer on count questions"
	python train.py -trainMode sl-abot -useGPU \
					-inputQues ../data/visdial_submodule/data/visdial_data.h5 \
					-inputJson ../data/visdial_submodule/data/visdial_params.json \
					-inputImg ../data/visdial_submodule/data/image_feats_res101.h5 \
					-cocoDir ../data/visdial_submodule/data/visdial_images \
					-cocoInfo ../data/visdial_submodule/data/visdial_images/coco_info.json \
					-categoryMap ../data/visdial_submodule/data/qa_category_mapping.json  \
					-imgNorm 0 \
					-imgFeatureSize 2048 \
					-enableVisdom 1 \
					-visdomServerPort 8895 \
					-visdomEnv count_v3 \
					-qaCategory count \
					-saveName count_v3

elif [ "$1" == 'debug' ]
then
	echo "Debugging Answerer training"
	python train.py -trainMode sl-abot -useGPU \
					-inputQues ../data/visdial_submodule/data/visdial_data.h5 \
					-inputJson ../data/visdial_submodule/data/visdial_params.json \
					-inputImg ../data/visdial_submodule/data/image_feats_res101.h5 \
					-cocoDir ../data/visdial_submodule/data/visdial_images \
					-cocoInfo ../data/visdial_submodule/data/visdial_images/coco_info.json \
					-categoryMap ../data/visdial_submodule/data/qa_category_mapping.json  \
					-imgNorm 0 \
					-imgFeatureSize 2048 \
					-enableVisdom 1 \
					-visdomEnv debug \
					-visdomServerPort 8895 \
					-saveName debug/visdom_debug

else
	echo "Invalid Category specification"

fi


