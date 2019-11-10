#!/bin/bash
# Usage: `./scripts/train_answerers.sh <training_mode> (<category>) (no_wait)`

wait_for_approval() {
        read  -n 1 -p "Enter [y] to confirm and proceed. Any other character to exit: " confirmation
        echo ""
        if [ $confirmation == "y" ]; then
                echo "Proceeding with training:"
                echo ""
        else
                exit 0
        fi
}

if [ "$1" == 'all_categories' ]; then
	echo "Training Answerer on all categories"
	python train.py -trainMode sl-abot -useGPU \
					-inputQues ../data/visdial_submodule/data/visdial_data_partition.h5 \
					-inputJson ../data/visdial_submodule/data/visdial_params_partition.json \
					-inputImg ../data/visdial_submodule/data/image_feats_res101_partition.h5 \
					-cocoDir ../data/visdial_submodule/data/visdial_images \
					-cocoInfo ../data/visdial_submodule/data/visdial_images/coco_info.json \
					-categoryMap ../data/visdial_submodule/data/qa_category_mapping.json  \
					-splitNames ../data/visdial_submodule/data/partition_split_names.json \
					-imgNorm 0 \
					-imgFeatureSize 2048 \
					-enableVisdom 1 \
					-visdomServerPort 8895 \
					-visdomEnv all_categories_v3 \
					-savePath checkpoints/all_categories \
					-saveName all_categories_v3 \
					-descr "Trained on all question categories."
	echo "Training finished!"

elif [ "$1" == 'category_turnwise_filter' ]; then
	echo "Training Answerer on category \"$2\". Dialogs filtered turnwise."
	if [ "$3" != "no_wait" ]; then wait_for_approval; fi
	python train.py -trainMode sl-abot -useGPU \
					-inputQues ../data/visdial_submodule/data/visdial_data_category_turnwise/visdial_data_$2.h5 \
					-inputJson ../data/visdial_submodule/data/visdial_params_category_turnwise/visdial_params_$2.json \
					-inputImg ../data/visdial_submodule/data/image_feats_res101_partition.h5 \
					-cocoDir ../data/visdial_submodule/data/visdial_images \
					-cocoInfo ../data/visdial_submodule/data/visdial_images/coco_info.json \
					-categoryMap ../data/visdial_submodule/data/qa_category_mapping.json  \
					-splitNames ../data/visdial_submodule/data/partition_split_names.json \
					-imgNorm 0 \
					-imgFeatureSize 2048 \
					-enableVisdom 1 \
					-visdomServerPort 8895 \
					-qaCategory $2 \
					-visdomEnv $2_v3_turnwise \
					-savePath checkpoints/$2 \
					-saveName $2_v3_turnwise \
					-descr "Trained on \"$2\" questions: only \"$2\" dialog turns included in dataset, filtered turnwise."
	echo "Training finished!"

elif [ "$1" == 'category_dialogwise_filter' ]; then
	echo "Training Answerer on category \"$2\". Dialogs filtered dialogwise."
	if [ "$3" != "no_wait" ]; then wait_for_approval; fi
	python train.py -trainMode sl-abot -useGPU \
					-inputQues ../data/visdial_submodule/data/visdial_data_category_dialogwise/visdial_data_$2.h5 \
					-inputJson ../data/visdial_submodule/data/visdial_params_category_dialogwise/visdial_params_$2.json \
					-inputImg ../data/visdial_submodule/data/image_feats_res101_partition.h5 \
					-cocoDir ../data/visdial_submodule/data/visdial_images \
					-cocoInfo ../data/visdial_submodule/data/visdial_images/coco_info.json \
					-categoryMap ../data/visdial_submodule/data/qa_category_mapping.json  \
					-splitNames ../data/visdial_submodule/data/partition_split_names.json \
					-imgNorm 0 \
					-imgFeatureSize 2048 \
					-enableVisdom 1 \
					-visdomServerPort 8895 \
					-qaCategory $2 \
					-visdomEnv $2_v3_dialogwise \
					-savePath checkpoints/$2 \
					-saveName $2_v3_dialogwise \
					-descr "Trained on \"$2\" questions: only \"$2\" dialog turns included in dataset, filtered dialogwise."
	echo "Training finished!"

elif [ "$1" == 'custom_script' ]; then
	echo "Selecting custom training scripts. Custom scripts further specified with arguments 2+"

	if [ "$2" == 'pretrain_vqa_all' ]; then
		python train.py -trainMode sl-abot -useGPU \
						-inputQues ../data/vqa/vqa_data.h5 \
						-inputJson ../data/vqa/vqa_params.json \
						-inputImg ../data/visdial_submodule/data/image_feats_res101_standard.h5 \
						-cocoDir ../data/visdial_submodule/data/visdial_images \
						-cocoInfo ../data/visdial_submodule/data/visdial_images/coco_info.json \
						-categoryMap ../data/visdial_submodule/data/qa_category_mapping.json  \
						-splitNames ../data/visdial_submodule/data/partition_split_names.json \
						-numRounds 1 \
						-noTestSet \
						-imgNorm 0 \
						-imgFeatureSize 2048 \
						-enableVisdom 1 \
						-visdomServerPort 8895 \
						-batchSize 10 \
						-visdomEnv all_categories_vqa \
						-savePath checkpoints/all_categories \
						-saveName all_categories_vqa \
						-descr "Trained on all question categories, from vqa data." \
						-clobberSave
	echo "Training finished!"

	elif [ "$2" == 'pretrain_vqa_category' ]; then
		echo "pretraining on vqa"
		python train.py -trainMode sl-abot -useGPU \
						-inputQues ../data/vqa/vqa_category_data/vqa_data_$3.h5 \
						-inputJson ../data/vqa/vqa_category_params/vqa_params_$3.json \
						-inputImg ../data/visdial_submodule/data/image_feats_res101_standard.h5 \
						-cocoDir ../data/vqa/images \
						-cocoInfo ../data/visdial_submodule/data/visdial_images/coco_info.json \
						-categoryMap ../data/visdial_submodule/data/qa_category_mapping.json  \
						-splitNames ../data/visdial_submodule/data/partition_split_names.json \
						-numRounds 1 \
						-noTestSet \
						-imgNorm 0 \
						-imgFeatureSize 2048 \
						-enableVisdom 1 \
						-visdomServerPort 8895 \
						-qaCategory $3 \
						-visdomEnv $3_vqa \
						-savePath checkpoints/$3 \
						-saveName $3_vqa \
						-descr "Trained on \"$3\" questions from vqa."
	echo "Training finished!"

	fi

	echo "Returning."
else
	echo "Invalid training mode in argument #1. Choose either \"all_categories\", \"category_turnwise_filter\", \"category_dialogwise_filter\", or \"custom_script\"."
fi


