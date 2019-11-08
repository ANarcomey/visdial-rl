#!/bin/bash
# Usage: `./scripts/eval_answerers_v2.sh <trained mode> <params>....`

wait_for_approval() {
        read  -n 1 -p "Enter [y] to confirm and proceed. Any other character to exit: " confirmation
        echo ""
        if [ $confirmation == "y" ]; then
                echo "Proceeding with evaluation:"
                echo ""
        else
                exit 0
        fi
}

if [ "$1" == 'trained_all_categories' ]; then
	echo "Evaluating Answerer trained on all categories."

	if [ "$2" == 'eval_on_category_turnwise' ]; then
		echo "Evaluating Answerer on category \"$3\". Dialogs filtered turnwise. Checkpoint model from \"$4\""
		if [ "$5" != "no_wait" ]; then wait_for_approval; fi
		python evaluate.py -useGPU -evalMode ABotRank \
						-inputQues ../data/visdial_submodule/data/visdial_partition_category_turnwise_v2/$3/data_vocab_from_all_categories.h5  \
						-inputJson ../data/visdial_submodule/data/visdial_partition_category_turnwise_v2/$3/params_vocab_from_all_categories.json \
						-inputImg ../data/visdial_submodule/data/image_feats_res101_partition.h5 \
						-cocoDir ../data/visdial_submodule/data/visdial_images \
						-cocoInfo ../data/visdial_submodule/data/visdial_images/coco_info.json \
						-categoryMap ../data/visdial_submodule/data/qa_category_mapping.json  \
						-splitNames ../data/visdial_submodule/data/partition_split_names.json \
						-imgNorm 0 \
						-imgFeatureSize 2048 \
						-enableVisdom 0 \
						-visdomServerPort 8895 \
						-qaCategory $3 \
						-visdomEnv eval_all_v3 \
						-savePath checkpoints_eval/all_categories_v3 \
						-saveName trained_all_eval_$3_turnwise \
						-startFrom $4 \
						-descr "Trained on all categories. Evaluated on $3 filtered turnwise."

	elif [ "$2" == 'eval_on_category_dialogwise' ]; then
		echo "Evaluating Answerer on category \"$3\". Dialogs filtered dialogwise. Checkpoint model from \"$4\""
		if [ "$5" != "no_wait" ]; then wait_for_approval; fi
		python evaluate.py -useGPU -evalMode ABotRank \
						-inputQues ../data/visdial_submodule/data/visdial_partition_category_dialogwise_v2/$3/data_vocab_from_all_categories.h5 \
						-inputJson ../data/visdial_submodule/data/visdial_partition_category_dialogwise_v2/$3/params_vocab_from_all_categories.json \
						-inputImg ../data/visdial_submodule/data/image_feats_res101_partition.h5 \
						-cocoDir ../data/visdial_submodule/data/visdial_images \
						-cocoInfo ../data/visdial_submodule/data/visdial_images/coco_info.json \
						-categoryMap ../data/visdial_submodule/data/qa_category_mapping.json  \
						-splitNames ../data/visdial_submodule/data/partition_split_names.json \
						-imgNorm 0 \
						-imgFeatureSize 2048 \
						-enableVisdom 0 \
						-visdomServerPort 8895 \
						-qaCategory $3 \
						-visdomEnv eval_all_v3 \
						-savePath checkpoints_eval/all_categories_v3 \
						-saveName trained_all_eval_$3_dialogwise \
						-startFrom $4 \
						-descr "Trained on all categories. Evaluated on $3 filtered dialogwise."

	elif [ "$2" == 'eval_on_each_category_turnwise' ]; then
		./scripts/eval_answerers_v2.sh trained_all_categories eval_on_category_turnwise binary $4 no_wait
		./scripts/eval_answerers_v2.sh trained_all_categories eval_on_category_turnwise color $4 no_wait
		./scripts/eval_answerers_v2.sh trained_all_categories eval_on_category_turnwise count $4 no_wait
		./scripts/eval_answerers_v2.sh trained_all_categories eval_on_category_turnwise object $4 no_wait
		./scripts/eval_answerers_v2.sh trained_all_categories eval_on_category_turnwise attribute $4 no_wait
		./scripts/eval_answerers_v2.sh trained_all_categories eval_on_category_turnwise predicate $4 no_wait
		./scripts/eval_answerers_v2.sh trained_all_categories eval_on_category_turnwise location $4 no_wait
		./scripts/eval_answerers_v2.sh trained_all_categories eval_on_category_turnwise time $4 no_wait
		./scripts/eval_answerers_v2.sh trained_all_categories eval_on_category_turnwise animal $4 no_wait
		./scripts/eval_answerers_v2.sh trained_all_categories eval_on_category_turnwise spatial $4 no_wait
		./scripts/eval_answerers_v2.sh trained_all_categories eval_on_category_turnwise material $4 no_wait
		./scripts/eval_answerers_v2.sh trained_all_categories eval_on_category_turnwise food $4 no_wait
		./scripts/eval_answerers_v2.sh trained_all_categories eval_on_category_turnwise activity $4 no_wait
		./scripts/eval_answerers_v2.sh trained_all_categories eval_on_category_turnwise stuff $4 no_wait
		./scripts/eval_answerers_v2.sh trained_all_categories eval_on_category_turnwise shape $4 no_wait

	elif [ "$2" == 'eval_on_each_category_dialogwise' ]; then
		./scripts/eval_answerers_v2.sh trained_all_categories eval_on_category_dialogwise binary $4 no_wait
		./scripts/eval_answerers_v2.sh trained_all_categories eval_on_category_dialogwise color $4 no_wait
		./scripts/eval_answerers_v2.sh trained_all_categories eval_on_category_dialogwise count $4 no_wait
		./scripts/eval_answerers_v2.sh trained_all_categories eval_on_category_dialogwise object $4 no_wait
		./scripts/eval_answerers_v2.sh trained_all_categories eval_on_category_dialogwise attribute $4 no_wait
		./scripts/eval_answerers_v2.sh trained_all_categories eval_on_category_dialogwise predicate $4 no_wait
		./scripts/eval_answerers_v2.sh trained_all_categories eval_on_category_dialogwise location $4 no_wait
		./scripts/eval_answerers_v2.sh trained_all_categories eval_on_category_dialogwise time $4 no_wait
		./scripts/eval_answerers_v2.sh trained_all_categories eval_on_category_dialogwise animal $4 no_wait
		./scripts/eval_answerers_v2.sh trained_all_categories eval_on_category_dialogwise spatial $4 no_wait
		./scripts/eval_answerers_v2.sh trained_all_categories eval_on_category_dialogwise material $4 no_wait
		./scripts/eval_answerers_v2.sh trained_all_categories eval_on_category_dialogwise food $4 no_wait
		./scripts/eval_answerers_v2.sh trained_all_categories eval_on_category_dialogwise activity $4 no_wait
		./scripts/eval_answerers_v2.sh trained_all_categories eval_on_category_dialogwise stuff $4 no_wait
		./scripts/eval_answerers_v2.sh trained_all_categories eval_on_category_dialogwise shape $4 no_wait

	elif [ "$2" == 'eval_on_all_categories' ]; then

		echo "Evaluating Answerer on complete dataset, all categories. Checkpoint model from \"$3\""
		python evaluate.py -useGPU -evalMode ABotRank \
						-inputQues ../data/visdial_submodule/data/visdial_partition_all_categories/data_vocab_from_all_categories.h5 \
						-inputJson ../data/visdial_submodule/data/visdial_partition_all_categories/params_vocab_from_all_categories.json \
						-inputQues ../data/visdial_submodule/data/visdial_data_partition.h5 \
						-inputJson ../data/visdial_submodule/data/visdial_params_partition.json \
						-inputImg ../data/visdial_submodule/data/image_feats_res101_partition.h5 \
						-cocoDir ../data/visdial_submodule/data/visdial_images \
						-cocoInfo ../data/visdial_submodule/data/visdial_images/coco_info.json \
						-categoryMap ../data/visdial_submodule/data/qa_category_mapping.json  \
						-splitNames ../data/visdial_submodule/data/partition_split_names.json \
						-imgNorm 0 \
						-imgFeatureSize 2048 \
						-enableVisdom 0 \
						-visdomServerPort 8895 \
						-visdomEnv eval_all_v3 \
						-savePath checkpoints_eval/all_categories_v3 \
						-saveName trained_all_eval_all \
						-startFrom $3 \
						-descr "Trained on all categories. Evaluated on all categories."

	else
		echo "Choose either \"eval_on_category_turnwise\" or \"eval_on_category_dialogwise\" for argument 2."
		echo "Alternatively, choose either \"eval_on_each_category_turnwise\" or \"eval_on_each_category_dialogwise\"" \
			 	"for argument 2, in order to evaluate for each of the categories."
		echo "Finally, choose \"eval_on_all_categories\" to evaluate on all data without any category filtering."
	fi

elif [ "$1" == 'trained_category_dialogwise' ]; then
	echo "Evaluating Answerer trained on \"$2\" category, filtered dialogwise. Checkpoint model from \"$3\"."

	if [ "$4" == 'eval_on_all_categories' ]; then
		echo "Evaluating on dialogs of all categories. No filtering."
		if [ "$5" != "no_wait" ]; then wait_for_approval; fi
		python evaluate.py -useGPU -evalMode ABotRank \
						-inputQues ../data/visdial_submodule/data/visdial_partition_all_categories/data_vocab_from_$2_dialogwise.h5 \
						-inputJson ../data/visdial_submodule/data/visdial_partition_all_categories/params_vocab_from_$2_dialogwise.json \
						-inputImg ../data/visdial_submodule/data/image_feats_res101_partition.h5 \
						-cocoDir ../data/visdial_submodule/data/visdial_images \
						-cocoInfo ../data/visdial_submodule/data/visdial_images/coco_info.json \
						-categoryMap ../data/visdial_submodule/data/qa_category_mapping.json  \
						-splitNames ../data/visdial_submodule/data/partition_split_names.json \
						-imgNorm 0 \
						-imgFeatureSize 2048 \
						-enableVisdom 0 \
						-visdomServerPort 8895 \
						-visdomEnv eval_$2_v3 \
						-savePath checkpoints_eval/$2_v3 \
						-saveName trained_$2_dialogwise_eval_all \
						-startFrom $3 \
						-descr "Trained on \"$2\" questions, filtered dialogwise. Evaluated on all categories" \


	elif [ "$4" == 'eval_on_category_turnwise' ]; then
		echo "Evaluating on dialogs of category \"$2\". Dialogs filtered turnwise"
		if [ "$5" != "no_wait" ]; then wait_for_approval; fi
		python evaluate.py -useGPU -evalMode ABotRank \
						-inputQues ../data/visdial_submodule/data/visdial_partition_category_turnwise_v2/$2/data_vocab_from_$2_dialogwise.h5 \
						-inputJson ../data/visdial_submodule/data/visdial_partition_category_turnwise_v2/$2/params_vocab_from_$2_dialogwise.json \
						-inputImg ../data/visdial_submodule/data/image_feats_res101_partition.h5 \
						-cocoDir ../data/visdial_submodule/data/visdial_images \
						-cocoInfo ../data/visdial_submodule/data/visdial_images/coco_info.json \
						-categoryMap ../data/visdial_submodule/data/qa_category_mapping.json  \
						-splitNames ../data/visdial_submodule/data/partition_split_names.json \
						-imgNorm 0 \
						-imgFeatureSize 2048 \
						-enableVisdom 0 \
						-visdomServerPort 8895 \
						-qaCategory $2 \
						-visdomEnv eval_$2_v3 \
						-savePath checkpoints_eval/$2_v3 \
						-saveName trained_$2_dialogwise_eval_$2_turnwise \
						-startFrom $3 \
						-descr "Trained on \"$2\" questions, filtered dialogwise. Evaluated on \"$2\" questions filtered turnwise."

	elif [ "$4" == 'eval_on_category_dialogwise' ]; then
		echo "Evaluating on dialogs of category \"$2\". Dialogs filtered dialogwise"
		if [ "$5" != "no_wait" ]; then wait_for_approval; fi
		python evaluate.py -useGPU -evalMode ABotRank \
						-inputQues ../data/visdial_submodule/data/visdial_partition_category_dialogwise_v2/$2/data_vocab_from_$2_dialogwise.h5 \
						-inputJson ../data/visdial_submodule/data/visdial_partition_category_dialogwise_v2/$2/params_vocab_from_$2_dialogwise.json \
						-inputImg ../data/visdial_submodule/data/image_feats_res101_partition.h5 \
						-cocoDir ../data/visdial_submodule/data/visdial_images \
						-cocoInfo ../data/visdial_submodule/data/visdial_images/coco_info.json \
						-categoryMap ../data/visdial_submodule/data/qa_category_mapping.json  \
						-splitNames ../data/visdial_submodule/data/partition_split_names.json \
						-imgNorm 0 \
						-imgFeatureSize 2048 \
						-enableVisdom 0 \
						-visdomServerPort 8895 \
						-qaCategory $2 \
						-visdomEnv eval_$2_v3 \
						-savePath checkpoints_eval/$2_v3 \
						-saveName trained_$2_dialogwise_eval_$2_dialogwise \
						-startFrom $3 \
						-descr "Trained on \"$2\" questions, filtered dialogwise. Evaluated on \"$2\" questions filtered dialogwise."

	elif [ "$4" == 'eval_all_modes' ]; then
		./scripts/eval_answerers_v2.sh trained_category_dialogwise $2 $3 eval_on_all_categories no_wait
		./scripts/eval_answerers_v2.sh trained_category_dialogwise $2 $3 eval_on_category_turnwise no_wait
		./scripts/eval_answerers_v2.sh trained_category_dialogwise $2 $3 eval_on_category_dialogwise no_wait
	else
		echo "Choose either \"eval_on_all_categories\", \"eval_on_category_turnwise\", or \"eval_on_category_dialogwise\" for argument 4."
		echo "Alternatively, choose \"eval_all_modes\" for argument 4 to run all these modes at once."
	fi

elif [ "$1" == 'trained_category_turnwise' ]; then
	echo "Evaluating Answerer trained on \"$2\" category, filtered turnwise. Checkpoint model from \"$3\"."

	if [ "$4" == 'eval_on_all_categories' ]; then
		echo "Evaluating on dialogs of all categories. No filtering."
		if [ "$5" != "no_wait" ]; then wait_for_approval; fi
		python evaluate.py -useGPU -evalMode ABotRank \
						-inputQues ../data/visdial_submodule/data/visdial_partition_all_categories/data_vocab_from_$2_turnwise.h5 \
						-inputJson ../data/visdial_submodule/data/visdial_partition_all_categories/params_vocab_from_$2_turnwise.json \
						-inputImg ../data/visdial_submodule/data/image_feats_res101_partition.h5 \
						-cocoDir ../data/visdial_submodule/data/visdial_images \
						-cocoInfo ../data/visdial_submodule/data/visdial_images/coco_info.json \
						-categoryMap ../data/visdial_submodule/data/qa_category_mapping.json  \
						-splitNames ../data/visdial_submodule/data/partition_split_names.json \
						-imgNorm 0 \
						-imgFeatureSize 2048 \
						-enableVisdom 0 \
						-visdomServerPort 8895 \
						-visdomEnv eval_$2_v3 \
						-savePath checkpoints_eval/$2_v3 \
						-saveName trained_$2_turnwise_eval_all \
						-startFrom $3 \
						-descr "Trained on \"$2\" questions, filtered turnwise. Evaluated on all categories"

	elif [ "$4" == 'eval_on_category_turnwise' ]; then

		echo "Evaluating on dialogs of category \"$2\". Dialogs filtered turnwise"
		if [ "$5" != "no_wait" ]; then wait_for_approval; fi
		python evaluate.py -useGPU -evalMode ABotRank \
						-inputQues ../data/visdial_submodule/data/visdial_partition_category_turnwise_v2/$2/data_vocab_from_$2_turnwise.h5 \
						-inputJson ../data/visdial_submodule/data/visdial_partition_category_turnwise_v2/$2/params_vocab_from_$2_turnwise.json \
						-inputImg ../data/visdial_submodule/data/image_feats_res101_partition.h5 \
						-cocoDir ../data/visdial_submodule/data/visdial_images \
						-cocoInfo ../data/visdial_submodule/data/visdial_images/coco_info.json \
						-categoryMap ../data/visdial_submodule/data/qa_category_mapping.json  \
						-splitNames ../data/visdial_submodule/data/partition_split_names.json \
						-imgNorm 0 \
						-imgFeatureSize 2048 \
						-enableVisdom 0 \
						-visdomServerPort 8895 \
						-qaCategory $2 \
						-visdomEnv eval_$2_v3 \
						-savePath checkpoints_eval/$2_v3 \
						-saveName trained_$2_turnwise_eval_$2_turnwise \
						-startFrom $3 \
						-descr "Trained on \"$2\" questions, filtered dialogwise. Evaluated on \"$2\" questions filtered turnwise."

	elif [ "$4" == 'eval_on_category_dialogwise' ]; then

		echo "Evaluating on dialogs of category \"$2\". Dialogs filtered dialogwise"
		if [ "$5" != "no_wait" ]; then wait_for_approval; fi
		python evaluate.py -useGPU -evalMode ABotRank \
						-inputQues ../data/visdial_submodule/data/visdial_partition_category_dialogwise_v2/$2/data_vocab_from_$2_turnwise.h5 \
						-inputJson ../data/visdial_submodule/data/visdial_partition_category_dialogwise_v2/$2/params_vocab_from_$2_turnwise.json \
						-inputImg ../data/visdial_submodule/data/image_feats_res101_partition.h5 \
						-cocoDir ../data/visdial_submodule/data/visdial_images \
						-cocoInfo ../data/visdial_submodule/data/visdial_images/coco_info.json \
						-categoryMap ../data/visdial_submodule/data/qa_category_mapping.json  \
						-splitNames ../data/visdial_submodule/data/partition_split_names.json \
						-imgNorm 0 \
						-imgFeatureSize 2048 \
						-enableVisdom 0 \
						-visdomServerPort 8895 \
						-qaCategory $2 \
						-visdomEnv eval_$2_v3 \
						-savePath checkpoints_eval/$2_v3 \
						-saveName trained_$2_turnwise_eval_$2_dialogwise \
						-startFrom $3 \
						-descr "Trained on \"$2\" questions, filtered turnwise. Evaluated on \"$2\" questions filtered dialogwise."


	elif [ "$4" == 'eval_all_modes' ]; then
		./scripts/eval_answerers_v2.sh trained_category_turnwise $2 $3 eval_on_all_categories no_wait
		./scripts/eval_answerers_v2.sh trained_category_turnwise $2 $3 eval_on_category_turnwise no_wait
		./scripts/eval_answerers_v2.sh trained_category_turnwise $2 $3 eval_on_category_dialogwise no_wait

	else
		echo "Choose either \"eval_on_all_categories\", \"eval_on_category_turnwise\", or \"eval_on_category_dialogwise\" for argument 4."
		echo "Alternatively, choose \"eval_all_modes\" for argument 4 to run all these modes at once."
	fi


elif [ "$1" == 'custom_scripts' ]; then
	echo "Selecting custom training scripts. Custom scripts further specified with arguments 2+"

	if [ "$2" == 'predicate_debug' ]; then
		echo "Running custom script for predicate_debug"
		python evaluate.py -useGPU -evalMode ABotRank \
							-inputQues ../data/visdial_submodule/data/visdial_data_predicate_with_test_gt.h5 \
							-inputJson ../data/visdial_submodule/data/visdial_params_predicate_with_test_gt.json \
							-inputImg ../data/visdial_submodule/data/image_feats_res101_partition.h5 \
							-cocoDir ../data/visdial_submodule/data/visdial_images \
							-cocoInfo ../data/visdial_submodule/data/visdial_images/coco_info.json \
							-categoryMap ../data/visdial_submodule/data/qa_category_mapping.json  \
							-splitNames ../data/visdial_submodule/data/partition_split_names.json \
							-imgNorm 0 \
							-imgFeatureSize 2048 \
							-enableVisdom 0 \
							-visdomServerPort 8895 \
							-visdomEnv eval_$2_v3 \
							-savePath checkpoints_eval/$2_v3 \
							-saveName trained_$2_dialogwise_eval_$2_dialogwise \
							-startFrom checkpoints_completed/$3 \
							-descr "Trained on \"$2\" questions, filtered dialogwise. Evaluated on \"$2\" questions filtered dialogwise."

	elif [ "$2" == 'eval_chunk1' ]; then
		echo "eval chunk 1"
		./scripts/eval_answerers_v2.sh trained_category_dialogwise color /vision2/u/anarc/motm/visdial-rl_submodule/checkpoints_completed/object/color_v3_dialogwise/abot_ep_64.vd eval_all_modes
		./scripts/eval_answerers_v2.sh trained_category_dialogwise object /vision2/u/anarc/motm/visdial-rl_submodule/checkpoints_completed/object/object_v3_dialogwise/abot_ep_64.vd eval_all_modes
		./scripts/eval_answerers_v2.sh trained_category_dialogwise attribute /vision2/u/anarc/motm/visdial-rl_submodule/checkpoints_completed/attribute/attribute_v3_dialogwise/abot_ep_64.vd eval_all_modes
		./scripts/eval_answerers_v2.sh trained_category_dialogwise predicate /vision2/u/anarc/motm/visdial-rl_submodule/checkpoints_completed/predicate/predicate_v3_dialogwise/abot_ep_64.vd eval_all_modes

	elif [ "$2" == 'eval_chunk2' ]; then
		echo "eval chunk 2"
		./scripts/eval_answerers_v2.sh trained_category_dialogwise location /vision2/u/anarc/motm/visdial-rl_submodule/checkpoints_completed/location/location_v3_dialogwise/abot_ep_64.vd eval_all_modes
		./scripts/eval_answerers_v2.sh trained_category_dialogwise time /vision2/u/anarc/motm/visdial-rl_submodule/checkpoints_completed/time/time_v3_dialogwise/abot_ep_64.vd eval_all_modes
		./scripts/eval_answerers_v2.sh trained_category_dialogwise animal /vision2/u/anarc/motm/visdial-rl_submodule/checkpoints_completed/animal/animal_v3_dialogwise/abot_ep_64.vd eval_all_modes
		./scripts/eval_answerers_v2.sh trained_category_dialogwise spatial /vision2/u/anarc/motm/visdial-rl_submodule/checkpoints_completed/spatial/spatial_v3_dialogwise/abot_ep_64.vd eval_all_modes

	elif [ "$2" == 'count' ]; then
		./scripts/eval_answerers_v2.sh trained_category_dialogwise count /vision2/u/anarc/motm/visdial-rl_submodule/checkpoints_completed/count/count_v3_dialogwise/abot_ep_64.vd eval_all_modes

	fi
	echo "Returning."

else
	echo "Invalid evaluation configuration in argument #1. Choose either \"trained_all_categories\", \"trained_category_turnwise\", \"trained_category_dialogwise\", or \"custom_script\"."
fi


