#!/bin/bash
# Usage: `./scripts/eval_answerers.sh color`

BASE_DIR=$(pwd)



if [ "$1" == 'all_categories' ]
then
	echo "Evaluate Answerer on all categories"

	python evaluate.py -useGPU \
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
					-savePath checkpoints_eval \
					-visdomEnv eval_all \
					-saveName all_newfeats \
					-evalMode ABotRank \
					-startFrom checkpoints/all_newfeats/abot_ep_50.vd


elif [ "$1" == 'color' ]
then
	echo "Evaluate Answerer on color category"

	python evaluate.py -useGPU \
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
					-savePath checkpoints_eval \
					-visdomEnv eval_color \
					-saveName color_newfeats \
					-evalMode ABotRank \
					-startFrom checkpoints/color_newfeats/abot_ep_50.vd 


else
	echo "Invalid Category specification"

fi


