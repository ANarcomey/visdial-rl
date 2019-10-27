#!/bin/bash
# Usage: `./scripts/train_answerers.sh color`

BASE_DIR=$(pwd)



if [ "$1" == 'all_categories' ]
then
	echo "Training Answerer on all categories"
	python train.py -trainMode sl-abot -useGPU \
					-inputQues ../data/visdial_submodule/data/visdial_data.h5 \
					-inputJson ../data/visdial_submodule/data/visdial_params.json \
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
					-saveName all_categories_v3

elif [ "$1" == 'binary' ]
then

	if [ "$2" == 'turnwise_filter' ]
	then
		echo "Training Answerer on binary questions, using turnwise category filtering"
		python train.py -trainMode sl-abot -useGPU \
						-inputQues ../data/visdial_submodule/data/visdial_data_category_turnwise/visdial_data_binary.h5 \
						-inputJson ../data/visdial_submodule/data/visdial_params_category_turnwise/visdial_params_binary.json \
						-inputImg ../data/visdial_submodule/data/image_feats_res101_partition.h5 \
						-cocoDir ../data/visdial_submodule/data/visdial_images \
						-cocoInfo ../data/visdial_submodule/data/visdial_images/coco_info.json \
						-categoryMap ../data/visdial_submodule/data/qa_category_mapping.json  \
						-splitNames ../data/visdial_submodule/data/partition_split_names.json \
						-imgNorm 0 \
						-imgFeatureSize 2048 \
						-enableVisdom 1 \
						-visdomServerPort 8895 \
						-visdomEnv binary_v3_turnwise \
						-savePath checkpoints/binary \
						-saveName binary_v3_turnwise \
						-descr "Trained on binary questions: only binary dialog turns included in dataset."

	elif [ "$2" == 'dialogwise_filter' ]
	then
		echo "Training Answerer on binary questions, using dialogwise category filtering"
		CUDA_LAUNCH_BLOCKING=1 python train.py -trainMode sl-abot -useGPU \
						-inputQues ../data/visdial_submodule/data/visdial_data_category_dialogwise/visdial_data_binary.h5 \
						-inputJson ../data/visdial_submodule/data/visdial_params_category_dialogwise/visdial_params_binary.json \
						-inputImg ../data/visdial_submodule/data/image_feats_res101_partition.h5 \
						-cocoDir ../data/visdial_submodule/data/visdial_images \
						-cocoInfo ../data/visdial_submodule/data/visdial_images/coco_info.json \
						-categoryMap ../data/visdial_submodule/data/qa_category_mapping.json  \
						-splitNames ../data/visdial_submodule/data/partition_split_names.json \
						-imgNorm 0 \
						-imgFeatureSize 2048 \
						-enableVisdom 1 \
						-visdomServerPort 8895 \
						-qaCategory binary \
						-visdomEnv binary_v3_dialogwise \
						-savePath checkpoints/binary \
						-saveName binary_v3_dialogwise \
						-descr "Trained on binary questions: complete dialog visible for context, loss only counted on binary dialog turns."

	elif [ "$2" == 'dialogwise_filter_debug' ]
	then
		echo "Training Answerer on binary questions, using dialogwise category filtering"
		CUDA_LAUNCH_BLOCKING=1 python train.py -trainMode sl-abot -useGPU \
						-inputQues ../data/visdial_submodule/data/visdial_data_category_dialogwise/visdial_data_binary.h5 \
						-inputJson ../data/visdial_submodule/data/visdial_params_category_dialogwise/visdial_params_binary.json \
						-inputImg ../data/visdial_submodule/data/image_feats_res101_partition.h5 \
						-cocoDir ../data/visdial_submodule/data/visdial_images \
						-cocoInfo ../data/visdial_submodule/data/visdial_images/coco_info.json \
						-categoryMap ../data/visdial_submodule/data/qa_category_mapping.json  \
						-splitNames ../data/visdial_submodule/data/partition_split_names.json \
						-imgNorm 0 \
						-imgFeatureSize 2048 \
						-enableVisdom 1 \
						-visdomServerPort 8895 \
						-qaCategory binary \
						-visdomEnv binary_v3_dialogwise_debug \
						-savePath checkpoints/binary_debug \
						-saveName binary_v3_dialogwise_debug \
						-descr "DEBUGGING: Trained on binary questions: complete dialog visible for context, loss only counted on binary dialog turns." \
						-clobberSave

	else
		echo "Choose either 'turnwise_filter' or 'dialogwise_filter' with argument #2"
	fi



elif [ "$1" == 'color' ]
then

	if [ "$2" == 'turnwise_filter' ]
	then
		echo "Training Answerer on color questions, using turnwise category filtering"
		python train.py -trainMode sl-abot -useGPU \
						-inputQues ../data/visdial_submodule/data/visdial_data_category_turnwise/visdial_data_color.h5 \
						-inputJson ../data/visdial_submodule/data/visdial_params_category_turnwise/visdial_params_color.json \
						-inputImg ../data/visdial_submodule/data/image_feats_res101_partition.h5 \
						-cocoDir ../data/visdial_submodule/data/visdial_images \
						-cocoInfo ../data/visdial_submodule/data/visdial_images/coco_info.json \
						-categoryMap ../data/visdial_submodule/data/qa_category_mapping.json  \
						-splitNames ../data/visdial_submodule/data/partition_split_names.json \
						-imgNorm 0 \
						-imgFeatureSize 2048 \
						-enableVisdom 1 \
						-visdomServerPort 8895 \
						-qaCategory color \
						-visdomEnv color_v3_turnwise \
						-savePath checkpoints/color \
						-saveName color_v3_turnwise \
						-descr "Trained on color questions: only color dialog turns included in dataset."

	elif [ "$2" == 'dialogwise_filter' ]
	then
		echo "Training Answerer on color questions, using dialogwise category filtering"
		python train.py -trainMode sl-abot -useGPU \
						-inputQues ../data/visdial_submodule/data/visdial_data_category_dialogwise/visdial_data_color.h5 \
						-inputJson ../data/visdial_submodule/data/visdial_params_category_dialogwise/visdial_params_color.json \
						-inputImg ../data/visdial_submodule/data/image_feats_res101_partition.h5 \
						-cocoDir ../data/visdial_submodule/data/visdial_images \
						-cocoInfo ../data/visdial_submodule/data/visdial_images/coco_info.json \
						-categoryMap ../data/visdial_submodule/data/qa_category_mapping.json  \
						-splitNames ../data/visdial_submodule/data/partition_split_names.json \
						-imgNorm 0 \
						-imgFeatureSize 2048 \
						-enableVisdom 1 \
						-visdomServerPort 8895 \
						-qaCategory color \
						-visdomEnv color_v3_dialogwise \
						-savePath checkpoints/color\
						-saveName color_v3_dialogwise \
						-descr "Trained on color questions: complete dialog visible for context, loss only counted on color dialog turns."

	elif [ "$2" == 'dialogwise_filter_debug' ]
	then
		echo "Training Answerer on color questions, using dialogwise category filtering"
		python train.py -trainMode sl-abot -useGPU \
						-inputQues ../data/visdial_submodule/data/visdial_data_category_dialogwise/visdial_data_color.h5 \
						-inputJson ../data/visdial_submodule/data/visdial_params_category_dialogwise/visdial_params_color.json \
						-inputImg ../data/visdial_submodule/data/image_feats_res101_partition.h5 \
						-cocoDir ../data/visdial_submodule/data/visdial_images \
						-cocoInfo ../data/visdial_submodule/data/visdial_images/coco_info.json \
						-categoryMap ../data/visdial_submodule/data/qa_category_mapping.json  \
						-splitNames ../data/visdial_submodule/data/partition_split_names.json \
						-imgNorm 0 \
						-imgFeatureSize 2048 \
						-enableVisdom 1 \
						-visdomServerPort 8895 \
						-qaCategory color \
						-visdomEnv color_v3_dialogwise_debug \
						-savePath checkpoints/color\
						-saveName color_v3_dialogwise_debug \
						-descr "Trained on color questions: complete dialog visible for context, loss only counted on color dialog turns." \
						-clobberSave

	else
		echo "Choose either 'turnwise_filter' or 'dialogwise_filter' with argument #2"
	fi
	

elif [ "$1" == 'count' ]
then
	if [ "$2" == 'turnwise_filter' ]
	then
		echo "Training Answerer on count questions, using turnwise category filtering"
		python train.py -trainMode sl-abot -useGPU \
						-inputQues ../data/visdial_submodule/data/visdial_data_category_turnwise/visdial_data_count.h5 \
						-inputJson ../data/visdial_submodule/data/visdial_params_category_turnwise/visdial_params_count.json \
						-inputImg ../data/visdial_submodule/data/image_feats_res101_partition.h5 \
						-cocoDir ../data/visdial_submodule/data/visdial_images \
						-cocoInfo ../data/visdial_submodule/data/visdial_images/coco_info.json \
						-categoryMap ../data/visdial_submodule/data/qa_category_mapping.json  \
						-splitNames ../data/visdial_submodule/data/partition_split_names.json \
						-imgNorm 0 \
						-imgFeatureSize 2048 \
						-enableVisdom 1 \
						-visdomServerPort 8895 \
						-qaCategory count \
						-visdomEnv count_v3_turnwise \
						-savePath checkpoints/count \
						-saveName count_v3_turnwise \
						-descr "Trained on count questions: only count dialog turns included in dataset."

	elif [ "$2" == 'dialogwise_filter' ]
	then
		echo "Training Answerer on count questions, using dialogwise category filtering"
		python train.py -trainMode sl-abot -useGPU \
						-inputQues ../data/visdial_submodule/data/visdial_data_category_dialogwise/visdial_data_count.h5 \
						-inputJson ../data/visdial_submodule/data/visdial_params_category_dialogwise/visdial_params_count.json \
						-inputImg ../data/visdial_submodule/data/image_feats_res101_partition.h5 \
						-cocoDir ../data/visdial_submodule/data/visdial_images \
						-cocoInfo ../data/visdial_submodule/data/visdial_images/coco_info.json \
						-categoryMap ../data/visdial_submodule/data/qa_category_mapping.json  \
						-splitNames ../data/visdial_submodule/data/partition_split_names.json \
						-imgNorm 0 \
						-imgFeatureSize 2048 \
						-enableVisdom 1 \
						-visdomServerPort 8895 \
						-qaCategory count \
						-visdomEnv count_v3_dialogwise \
						-savePath checkpoints/count\
						-saveName count_v3_dialogwise \
						-descr "Trained on count questions: complete dialog visible for context, loss only counted on count dialog turns."

	elif [ "$2" == 'dialogwise_filter_debug' ]
	then
		echo "Training Answerer on count questions, using dialogwise category filtering"
		python train.py -trainMode sl-abot -useGPU \
						-inputQues ../data/visdial_submodule/data/visdial_data_category_dialogwise/visdial_data_count.h5 \
						-inputJson ../data/visdial_submodule/data/visdial_params_category_dialogwise/visdial_params_count.json \
						-inputImg ../data/visdial_submodule/data/image_feats_res101_partition.h5 \
						-cocoDir ../data/visdial_submodule/data/visdial_images \
						-cocoInfo ../data/visdial_submodule/data/visdial_images/coco_info.json \
						-categoryMap ../data/visdial_submodule/data/qa_category_mapping.json  \
						-splitNames ../data/visdial_submodule/data/partition_split_names.json \
						-imgNorm 0 \
						-imgFeatureSize 2048 \
						-enableVisdom 1 \
						-visdomServerPort 8895 \
						-qaCategory count \
						-visdomEnv count_v3_dialogwise_debug \
						-savePath checkpoints/count\
						-saveName count_v3_dialogwise_debug \
						-descr "Trained on count questions: complete dialog visible for context, loss only counted on count dialog turns." \
						-clobberSave

	else
		echo "Choose either 'turnwise_filter' or 'dialogwise_filter' with argument #2"
	fi


elif [ "$1" == 'object' ]
then
	if [ "$2" == 'turnwise_filter' ]
	then
		echo "Training Answerer on object questions, using turnwise category filtering"
		python train.py -trainMode sl-abot -useGPU \
						-inputQues ../data/visdial_submodule/data/visdial_data_category_turnwise/visdial_data_object.h5 \
						-inputJson ../data/visdial_submodule/data/visdial_params_category_turnwise/visdial_params_object.json \
						-inputImg ../data/visdial_submodule/data/image_feats_res101_partition.h5 \
						-cocoDir ../data/visdial_submodule/data/visdial_images \
						-cocoInfo ../data/visdial_submodule/data/visdial_images/coco_info.json \
						-categoryMap ../data/visdial_submodule/data/qa_category_mapping.json  \
						-splitNames ../data/visdial_submodule/data/partition_split_names.json \
						-imgNorm 0 \
						-imgFeatureSize 2048 \
						-enableVisdom 1 \
						-visdomServerPort 8895 \
						-qaCategory object \
						-visdomEnv object_v3_turnwise \
						-savePath checkpoints/object \
						-saveName object_v3_turnwise \
						-descr "Trained on object questions: only object dialog turns included in dataset."

	elif [ "$2" == 'dialogwise_filter' ]
	then
		echo "Training Answerer on object questions, using dialogwise category filtering"
		python train.py -trainMode sl-abot -useGPU \
						-inputQues ../data/visdial_submodule/data/visdial_data_category_dialogwise/visdial_data_object.h5 \
						-inputJson ../data/visdial_submodule/data/visdial_params_category_dialogwise/visdial_params_object.json \
						-inputImg ../data/visdial_submodule/data/image_feats_res101_partition.h5 \
						-cocoDir ../data/visdial_submodule/data/visdial_images \
						-cocoInfo ../data/visdial_submodule/data/visdial_images/coco_info.json \
						-categoryMap ../data/visdial_submodule/data/qa_category_mapping.json  \
						-splitNames ../data/visdial_submodule/data/partition_split_names.json \
						-imgNorm 0 \
						-imgFeatureSize 2048 \
						-enableVisdom 1 \
						-visdomServerPort 8895 \
						-qaCategory object \
						-visdomEnv object_v3_dialogwise \
						-savePath checkpoints/object\
						-saveName object_v3_dialogwise \
						-descr "Trained on object questions: complete dialog visible for context, loss only counted on object dialog turns."

	else
		echo "Choose either 'turnwise_filter' or 'dialogwise_filter' with argument #2"
	fi

elif [ "$1" == 'attribute' ]
then
	if [ "$2" == 'turnwise_filter' ]
	then
		echo "Training Answerer on attribute questions, using turnwise category filtering"
		python train.py -trainMode sl-abot -useGPU \
						-inputQues ../data/visdial_submodule/data/visdial_data_category_turnwise/visdial_data_attribute.h5 \
						-inputJson ../data/visdial_submodule/data/visdial_params_category_turnwise/visdial_params_attribute.json \
						-inputImg ../data/visdial_submodule/data/image_feats_res101_partition.h5 \
						-cocoDir ../data/visdial_submodule/data/visdial_images \
						-cocoInfo ../data/visdial_submodule/data/visdial_images/coco_info.json \
						-categoryMap ../data/visdial_submodule/data/qa_category_mapping.json  \
						-splitNames ../data/visdial_submodule/data/partition_split_names.json \
						-imgNorm 0 \
						-imgFeatureSize 2048 \
						-enableVisdom 1 \
						-visdomServerPort 8895 \
						-qaCategory attribute \
						-visdomEnv attribute_v3_turnwise \
						-savePath checkpoints/attribute \
						-saveName attribute_v3_turnwise \
						-descr "Trained on attribute questions: only attribute dialog turns included in dataset."

	elif [ "$2" == 'dialogwise_filter' ]
	then
		echo "Training Answerer on attribute questions, using dialogwise category filtering"
		python train.py -trainMode sl-abot -useGPU \
						-inputQues ../data/visdial_submodule/data/visdial_data_category_dialogwise/visdial_data_attribute.h5 \
						-inputJson ../data/visdial_submodule/data/visdial_params_category_dialogwise/visdial_params_attribute.json \
						-inputImg ../data/visdial_submodule/data/image_feats_res101_partition.h5 \
						-cocoDir ../data/visdial_submodule/data/visdial_images \
						-cocoInfo ../data/visdial_submodule/data/visdial_images/coco_info.json \
						-categoryMap ../data/visdial_submodule/data/qa_category_mapping.json  \
						-splitNames ../data/visdial_submodule/data/partition_split_names.json \
						-imgNorm 0 \
						-imgFeatureSize 2048 \
						-enableVisdom 1 \
						-visdomServerPort 8895 \
						-qaCategory attribute \
						-visdomEnv attribute_v3_dialogwise \
						-savePath checkpoints/attribute\
						-saveName attribute_v3_dialogwise \
						-descr "Trained on attribute questions: complete dialog visible for context, loss only counted on attribute dialog turns."

	else
		echo "Choose either 'turnwise_filter' or 'dialogwise_filter' with argument #2"
	fi

elif [ "$1" == 'predicate' ]
then
	if [ "$2" == 'turnwise_filter' ]
	then
		echo "Training Answerer on predicate questions, using turnwise category filtering"
		python train.py -trainMode sl-abot -useGPU \
						-inputQues ../data/visdial_submodule/data/visdial_data_category_turnwise/visdial_data_predicate.h5 \
						-inputJson ../data/visdial_submodule/data/visdial_params_category_turnwise/visdial_params_predicate.json \
						-inputImg ../data/visdial_submodule/data/image_feats_res101_partition.h5 \
						-cocoDir ../data/visdial_submodule/data/visdial_images \
						-cocoInfo ../data/visdial_submodule/data/visdial_images/coco_info.json \
						-categoryMap ../data/visdial_submodule/data/qa_category_mapping.json  \
						-splitNames ../data/visdial_submodule/data/partition_split_names.json \
						-imgNorm 0 \
						-imgFeatureSize 2048 \
						-enableVisdom 1 \
						-visdomServerPort 8895 \
						-qaCategory predicate \
						-visdomEnv predicate_v3_turnwise \
						-savePath checkpoints/predicate \
						-saveName predicate_v3_turnwise \
						-descr "Trained on predicate questions: only predicate dialog turns included in dataset."

	elif [ "$2" == 'dialogwise_filter' ]
	then
		echo "Training Answerer on predicate questions, using dialogwise category filtering"
		python train.py -trainMode sl-abot -useGPU \
						-inputQues ../data/visdial_submodule/data/visdial_data_category_dialogwise/visdial_data_predicate.h5 \
						-inputJson ../data/visdial_submodule/data/visdial_params_category_dialogwise/visdial_params_predicate.json \
						-inputImg ../data/visdial_submodule/data/image_feats_res101_partition.h5 \
						-cocoDir ../data/visdial_submodule/data/visdial_images \
						-cocoInfo ../data/visdial_submodule/data/visdial_images/coco_info.json \
						-categoryMap ../data/visdial_submodule/data/qa_category_mapping.json  \
						-splitNames ../data/visdial_submodule/data/partition_split_names.json \
						-imgNorm 0 \
						-imgFeatureSize 2048 \
						-enableVisdom 1 \
						-visdomServerPort 8895 \
						-qaCategory predicate \
						-visdomEnv predicate_v3_dialogwise \
						-savePath checkpoints/predicate\
						-saveName predicate_v3_dialogwise \
						-descr "Trained on predicate questions: complete dialog visible for context, loss only counted on predicate dialog turns."

	else
		echo "Choose either 'turnwise_filter' or 'dialogwise_filter' with argument #2"
	fi

elif [ "$1" == 'location' ]
then
	if [ "$2" == 'turnwise_filter' ]
	then
		echo "Training Answerer on location questions, using turnwise category filtering"
		python train.py -trainMode sl-abot -useGPU \
						-inputQues ../data/visdial_submodule/data/visdial_data_category_turnwise/visdial_data_location.h5 \
						-inputJson ../data/visdial_submodule/data/visdial_params_category_turnwise/visdial_params_location.json \
						-inputImg ../data/visdial_submodule/data/image_feats_res101_partition.h5 \
						-cocoDir ../data/visdial_submodule/data/visdial_images \
						-cocoInfo ../data/visdial_submodule/data/visdial_images/coco_info.json \
						-categoryMap ../data/visdial_submodule/data/qa_category_mapping.json  \
						-splitNames ../data/visdial_submodule/data/partition_split_names.json \
						-imgNorm 0 \
						-imgFeatureSize 2048 \
						-enableVisdom 1 \
						-visdomServerPort 8895 \
						-qaCategory location \
						-visdomEnv location_v3_turnwise \
						-savePath checkpoints/location \
						-saveName location_v3_turnwise \
						-descr "Trained on location questions: only location dialog turns included in dataset."

	elif [ "$2" == 'dialogwise_filter' ]
	then
		echo "Training Answerer on location questions, using dialogwise category filtering"
		python train.py -trainMode sl-abot -useGPU \
						-inputQues ../data/visdial_submodule/data/visdial_data_category_dialogwise/visdial_data_location.h5 \
						-inputJson ../data/visdial_submodule/data/visdial_params_category_dialogwise/visdial_params_location.json \
						-inputImg ../data/visdial_submodule/data/image_feats_res101_partition.h5 \
						-cocoDir ../data/visdial_submodule/data/visdial_images \
						-cocoInfo ../data/visdial_submodule/data/visdial_images/coco_info.json \
						-categoryMap ../data/visdial_submodule/data/qa_category_mapping.json  \
						-imgNorm 0 \
						-imgFeatureSize 2048 \
						-enableVisdom 1 \
						-visdomServerPort 8895 \
						-qaCategory location \
						-visdomEnv location_v3_dialogwise \
						-savePath checkpoints/location\
						-saveName location_v3_dialogwise \
						-descr "Trained on location questions: complete dialog visible for context, loss only counted on location dialog turns."

	else
		echo "Choose either 'turnwise_filter' or 'dialogwise_filter' with argument #2"
	fi

elif [ "$1" == 'time' ]
then
	if [ "$2" == 'turnwise_filter' ]
	then
		echo "Training Answerer on time questions, using turnwise category filtering"
		python train.py -trainMode sl-abot -useGPU \
						-inputQues ../data/visdial_submodule/data/visdial_data_category_turnwise/visdial_data_time.h5 \
						-inputJson ../data/visdial_submodule/data/visdial_params_category_turnwise/visdial_params_time.json \
						-inputImg ../data/visdial_submodule/data/image_feats_res101_partition.h5 \
						-cocoDir ../data/visdial_submodule/data/visdial_images \
						-cocoInfo ../data/visdial_submodule/data/visdial_images/coco_info.json \
						-categoryMap ../data/visdial_submodule/data/qa_category_mapping.json  \
						-splitNames ../data/visdial_submodule/data/partition_split_names.json \
						-imgNorm 0 \
						-imgFeatureSize 2048 \
						-enableVisdom 1 \
						-visdomServerPort 8895 \
						-qaCategory time \
						-visdomEnv time_v3_turnwise \
						-savePath checkpoints/time \
						-saveName time_v3_turnwise \
						-descr "Trained on time questions: only time dialog turns included in dataset."

	elif [ "$2" == 'dialogwise_filter' ]
	then
		echo "Training Answerer on time questions, using dialogwise category filtering"
		python train.py -trainMode sl-abot -useGPU \
						-inputQues ../data/visdial_submodule/data/visdial_data_category_dialogwise/visdial_data_time.h5 \
						-inputJson ../data/visdial_submodule/data/visdial_params_category_dialogwise/visdial_params_time.json \
						-inputImg ../data/visdial_submodule/data/image_feats_res101_partition.h5 \
						-cocoDir ../data/visdial_submodule/data/visdial_images \
						-cocoInfo ../data/visdial_submodule/data/visdial_images/coco_info.json \
						-categoryMap ../data/visdial_submodule/data/qa_category_mapping.json  \
						-splitNames ../data/visdial_submodule/data/partition_split_names.json \
						-imgNorm 0 \
						-imgFeatureSize 2048 \
						-enableVisdom 1 \
						-visdomServerPort 8895 \
						-qaCategory time \
						-visdomEnv time_v3_dialogwise \
						-savePath checkpoints/time\
						-saveName time_v3_dialogwise \
						-descr "Trained on time questions: complete dialog visible for context, loss only counted on time dialog turns."

	else
		echo "Choose either 'turnwise_filter' or 'dialogwise_filter' with argument #2"
	fi

elif [ "$1" == 'debug' ]
then
	echo "Debugging Answerer training"
	python train.py -trainMode sl-abot -useGPU \
					-inputQues ../data/visdial_submodule/data/visdial_data.h5 \
					-inputJson ../data/visdial_submodule/data/visdial_params.json \
					-inputImg ../data/visdial_submodule/data/image_feats_res101_partition.h5 \
					-cocoDir ../data/visdial_submodule/data/visdial_images \
					-cocoInfo ../data/visdial_submodule/data/visdial_images/coco_info.json \
					-categoryMap ../data/visdial_submodule/data/qa_category_mapping.json  \
					-splitNames ../data/visdial_submodule/data/partition_split_names.json \
					-imgNorm 0 \
					-imgFeatureSize 2048 \
					-enableVisdom 1 \
					-visdomEnv debug \
					-visdomServerPort 8895 \
					-saveName debug/visdom_debug

else
	echo "Invalid Category specification"

fi


