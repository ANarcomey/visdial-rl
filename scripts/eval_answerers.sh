#!/bin/bash
# Usage: `./scripts/eval_answerers.sh color`

BASE_DIR=$(pwd)

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

if [ "$1" == 'all_categories' ]; then
	echo "Evaluating Answerer trained on all categories."

	if [ "$2" == 'eval_on_category_turnwise' ]; then

		echo "Evaluating Answerer on category \"$3\". Dialogs filtered turnwise."
		if [ "$4" != "no_wait" ]; then wait_for_approval; fi
		python evaluate.py -useGPU -evalMode ABotRank \
						-inputQues ../data/visdial_submodule/data/visdial_data_category_turnwise/visdial_data_$3.h5 \
						-inputJson ../data/visdial_submodule/data/visdial_params_category_turnwise/visdial_params_$3.json \
						-inputImg ../data/visdial_submodule/data/image_feats_res101_partition.h5 \
						-cocoDir ../data/visdial_submodule/data/visdial_images \
						-cocoInfo ../data/visdial_submodule/data/visdial_images/coco_info.json \
						-categoryMap ../data/visdial_submodule/data/qa_category_mapping.json  \
						-splitNames ../data/visdial_submodule/data/partition_split_names.json \
						-imgNorm 0 \
						-imgFeatureSize 2048 \
						-enableVisdom 1 \
						-visdomServerPort 8895 \
						-visdomEnv eval_all_v3 \
						-savePath checkpoints_eval/all_categories_v3 \
						-saveName trained_all_eval_$3_turnwise \
						-startFrom checkpoints/all_newfeats/abot_ep_50.vd \
						-descr "Trained on all categories. Evaluated on $3 filtered turnwise."

	elif [ "$2" == 'eval_on_category_dialogwise' ]; then

		echo "Evaluating Answerer on category \"$3\". Dialogs filtered dialogwise."
		if [ "$4" != "no_wait" ]; then wait_for_approval; fi
		python evaluate.py -useGPU -evalMode ABotRank \
						-inputQues ../data/visdial_submodule/data/visdial_data_category_dialogwise/visdial_data_$3.h5 \
						-inputJson ../data/visdial_submodule/data/visdial_params_category_dialogwise/visdial_params_$3.json \
						-inputImg ../data/visdial_submodule/data/image_feats_res101_partition.h5 \
						-cocoDir ../data/visdial_submodule/data/visdial_images \
						-cocoInfo ../data/visdial_submodule/data/visdial_images/coco_info.json \
						-categoryMap ../data/visdial_submodule/data/qa_category_mapping.json  \
						-splitNames ../data/visdial_submodule/data/partition_split_names.json \
						-imgNorm 0 \
						-imgFeatureSize 2048 \
						-enableVisdom 1 \
						-visdomServerPort 8895 \
						-visdomEnv eval_all_v3 \
						-savePath checkpoints_eval/all_categories_v3 \
						-saveName trained_all_eval_$3_dialogwise \
						-startFrom checkpoints/all_newfeats/abot_ep_50.vd \
						-descr "Trained on all categories. Evaluated on $3 filtered dialogwise."

	elif [ "$2" == 'eval_on_each_category_turnwise' ]; then
		./scripts/eval_answerers.sh all_categories eval_on_category_turnwise binary no_wait
		./scripts/eval_answerers.sh all_categories eval_on_category_turnwise color no_wait
		./scripts/eval_answerers.sh all_categories eval_on_category_turnwise count no_wait
		./scripts/eval_answerers.sh all_categories eval_on_category_turnwise object no_wait
		./scripts/eval_answerers.sh all_categories eval_on_category_turnwise attribute no_wait
		./scripts/eval_answerers.sh all_categories eval_on_category_turnwise predicate no_wait
		./scripts/eval_answerers.sh all_categories eval_on_category_turnwise location no_wait
		./scripts/eval_answerers.sh all_categories eval_on_category_turnwise time no_wait
		./scripts/eval_answerers.sh all_categories eval_on_category_turnwise animal no_wait
		./scripts/eval_answerers.sh all_categories eval_on_category_turnwise spatial no_wait
		./scripts/eval_answerers.sh all_categories eval_on_category_turnwise material no_wait
		./scripts/eval_answerers.sh all_categories eval_on_category_turnwise food no_wait
		./scripts/eval_answerers.sh all_categories eval_on_category_turnwise activity no_wait
		./scripts/eval_answerers.sh all_categories eval_on_category_turnwise stuff no_wait
		./scripts/eval_answerers.sh all_categories eval_on_category_turnwise shape no_wait

	elif [ "$2" == 'eval_on_each_category_dialogwise' ]; then
		./scripts/eval_answerers.sh all_categories eval_on_category_dialogwise binary no_wait
		./scripts/eval_answerers.sh all_categories eval_on_category_dialogwise color no_wait
		./scripts/eval_answerers.sh all_categories eval_on_category_dialogwise count no_wait
		./scripts/eval_answerers.sh all_categories eval_on_category_dialogwise object no_wait
		./scripts/eval_answerers.sh all_categories eval_on_category_dialogwise attribute no_wait
		./scripts/eval_answerers.sh all_categories eval_on_category_dialogwise predicate no_wait
		./scripts/eval_answerers.sh all_categories eval_on_category_dialogwise location no_wait
		./scripts/eval_answerers.sh all_categories eval_on_category_dialogwise time no_wait
		./scripts/eval_answerers.sh all_categories eval_on_category_dialogwise animal no_wait
		./scripts/eval_answerers.sh all_categories eval_on_category_dialogwise spatial no_wait
		./scripts/eval_answerers.sh all_categories eval_on_category_dialogwise material no_wait
		./scripts/eval_answerers.sh all_categories eval_on_category_dialogwise food no_wait
		./scripts/eval_answerers.sh all_categories eval_on_category_dialogwise activity no_wait
		./scripts/eval_answerers.sh all_categories eval_on_category_dialogwise stuff no_wait
		./scripts/eval_answerers.sh all_categories eval_on_category_dialogwise shape no_wait

	elif [ "$2" == 'eval_on_all_categories' ]; then

		echo "Evaluating Answerer on complete dataset, all categories."
		python evaluate.py -useGPU -evalMode ABotRank \
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
						-visdomEnv eval_all_v3 \
						-savePath checkpoints_eval/all_categories_v3 \
						-saveName trained_all_eval_all \
						-startFrom checkpoints/all_newfeats/abot_ep_50.vd \
						-descr "Trained on all categories. Evaluated on all categories."

	else
		echo "Choose either \"eval_on_category_turnwise\" or \"eval_on_category_dialogwise\" for argument 2."
		echo "Alternatively, choose either \"eval_on_each_category_turnwise\" or \"eval_on_each_category_dialogwise\"" \
			 	"for argument 2, in order to evaluate for each of the categories."
		echo "Finally, choose \"eval_on_all_categories\" to evaluate on all data without any category filtering."
	fi

elif [ "$1" == 'predicate_debug' ]; then
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
						-enableVisdom 1 \
						-visdomServerPort 8895 \
						-visdomEnv eval_$1_v3 \
						-savePath checkpoints_eval/$1_v3 \
						-saveName trained_$1_dialogwise_eval_$1_dialogwise \
						-startFrom checkpoints_completed/$2 \
						-descr "Trained on \"$1\" questions, filtered dialogwise. Evaluated on \"$1\" questions filtered dialogwise."

else
	echo "Evaluate Answerer trained on \"$1\" category. Checkpoint model from \"checkpoints_completed/$2\"."

	if [ "$3" == 'trained_dialogwise' ]; then
		echo "Evaluating Answerer trained on \"$1\" category, filtered dialogwise"

		if [ "$4" == 'eval_on_all_categories' ]; then
			echo "Evaluating on dialogs of all categories. No filtering."
			if [ "$5" != "no_wait" ]; then wait_for_approval; fi
			python evaluate.py -useGPU -evalMode ABotRank \
							-inputQues ../data/visdial_submodule/data/visdial_data_partition_predicate_vocab.h5 \
							-inputJson ../data/visdial_submodule/data/visdial_params_partition_predicate_vocab.json \
							-inputImg ../data/visdial_submodule/data/image_feats_res101_partition.h5 \
							-cocoDir ../data/visdial_submodule/data/visdial_images \
							-cocoInfo ../data/visdial_submodule/data/visdial_images/coco_info.json \
							-categoryMap ../data/visdial_submodule/data/qa_category_mapping.json  \
							-splitNames ../data/visdial_submodule/data/partition_split_names.json \
							-imgNorm 0 \
							-imgFeatureSize 2048 \
							-enableVisdom 1 \
							-visdomServerPort 8895 \
							-visdomEnv eval_$1_v3 \
							-savePath checkpoints_eval/$1_v3 \
							-saveName trained_$1_dialogwise_eval_all \
							-startFrom checkpoints_completed/$2 \
							-descr "Trained on \"$1\" questions, filtered dialogwise. Evaluated on all categories" \

			#-inputQues ../data/visdial_submodule/data/visdial_data_partition.h5 \
			#-inputJson ../data/visdial_submodule/data/visdial_params_partition.json \


		elif [ "$4" == 'eval_on_category_turnwise' ]; then
			echo "Evaluating on dialogs of category \"$1\". Dialogs filtered turnwise"
			if [ "$5" != "no_wait" ]; then wait_for_approval; fi
			python evaluate.py -useGPU -evalMode ABotRank \
							-inputQues ../data/visdial_submodule/data/visdial_data_category_turnwise/visdial_data_$1.h5 \
							-inputJson ../data/visdial_submodule/data/visdial_params_category_turnwise/visdial_params_$1.json \
							-inputImg ../data/visdial_submodule/data/image_feats_res101_partition.h5 \
							-cocoDir ../data/visdial_submodule/data/visdial_images \
							-cocoInfo ../data/visdial_submodule/data/visdial_images/coco_info.json \
							-categoryMap ../data/visdial_submodule/data/qa_category_mapping.json  \
							-splitNames ../data/visdial_submodule/data/partition_split_names.json \
							-imgNorm 0 \
							-imgFeatureSize 2048 \
							-enableVisdom 1 \
							-visdomServerPort 8895 \
							-visdomEnv eval_$1_v3 \
							-savePath checkpoints_eval/$1_v3 \
							-saveName trained_$1_dialogwise_eval_$1_turnwise \
							-startFrom checkpoints_completed/$2 \
							-descr "Trained on \"$1\" questions, filtered dialogwise. Evaluated on \"$1\" questions filtered turnwise."

		elif [ "$4" == 'eval_on_category_dialogwise' ]; then
			echo "Evaluating on dialogs of category \"$1\". Dialogs filtered dialogwise"
			if [ "$5" != "no_wait" ]; then wait_for_approval; fi
			python evaluate.py -useGPU -evalMode ABotRank \
							-inputQues ../data/visdial_submodule/data/visdial_data_category_dialogwise/visdial_data_$1.h5 \
							-inputJson ../data/visdial_submodule/data/visdial_params_category_dialogwise/visdial_params_$1.json \
							-inputImg ../data/visdial_submodule/data/image_feats_res101_partition.h5 \
							-cocoDir ../data/visdial_submodule/data/visdial_images \
							-cocoInfo ../data/visdial_submodule/data/visdial_images/coco_info.json \
							-categoryMap ../data/visdial_submodule/data/qa_category_mapping.json  \
							-splitNames ../data/visdial_submodule/data/partition_split_names.json \
							-imgNorm 0 \
							-imgFeatureSize 2048 \
							-enableVisdom 1 \
							-visdomServerPort 8895 \
							-visdomEnv eval_$1_v3 \
							-savePath checkpoints_eval/$1_v3 \
							-saveName trained_$1_dialogwise_eval_$1_dialogwise \
							-startFrom checkpoints_completed/$2 \
							-descr "Trained on \"$1\" questions, filtered dialogwise. Evaluated on \"$1\" questions filtered dialogwise."

		elif [ "$4" == 'eval_all_modes' ]; then
			./scripts/eval_answerers.sh $1 $2 trained_dialogwise eval_on_all_categories no_wait
			./scripts/eval_answerers.sh $1 $2 trained_dialogwise eval_on_category_turnwise no_wait
			./scripts/eval_answerers.sh $1 $2 trained_dialogwise eval_on_category_dialogwise no_wait

		else
			echo "Choose either \"eval_on_all_categories\", \"eval_on_category_turnwise\", or \"eval_on_category_dialogwise\" for argument 4."
			echo "Alternatively, choose \"eval_all_modes\" for argument 4 to run all these modes at once."
		fi

	elif [ "$3" == 'trained_turnwise' ]; then

		echo "Evaluating Answerer trained on \"$1\" category, filtered turnwise"

		if [ "$4" == 'eval_on_all_categories' ]; then
			echo "Evaluating on dialogs of all categories. No filtering."
			if [ "$5" != "no_wait" ]; then wait_for_approval; fi
			python evaluate.py -useGPU -evalMode ABotRank \
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
							-visdomEnv eval_$1_v3 \
							-savePath checkpoints_eval/$1_v3 \
							-saveName trained_$1_turnwise_eval_all \
							-startFrom checkpoints_completed/$2 \
							-descr "Trained on \"$1\" questions, filtered turnwise. Evaluated on all categories"

		elif [ "$4" == 'eval_on_category_turnwise' ]; then

			echo "Evaluating on dialogs of category \"$1\". Dialogs filtered turnwise"
			if [ "$5" != "no_wait" ]; then wait_for_approval; fi
			python evaluate.py -useGPU -evalMode ABotRank \
							-inputQues ../data/visdial_submodule/data/visdial_data_category_turnwise/visdial_data_$1.h5 \
							-inputJson ../data/visdial_submodule/data/visdial_params_category_turnwise/visdial_params_$1.json \
							-inputImg ../data/visdial_submodule/data/image_feats_res101_partition.h5 \
							-cocoDir ../data/visdial_submodule/data/visdial_images \
							-cocoInfo ../data/visdial_submodule/data/visdial_images/coco_info.json \
							-categoryMap ../data/visdial_submodule/data/qa_category_mapping.json  \
							-splitNames ../data/visdial_submodule/data/partition_split_names.json \
							-imgNorm 0 \
							-imgFeatureSize 2048 \
							-enableVisdom 1 \
							-visdomServerPort 8895 \
							-visdomEnv eval_$1_v3 \
							-savePath checkpoints_eval/$1_v3 \
							-saveName trained_$1_dialogwise_eval_$1_turnwise \
							-startFrom checkpoints_completed/$2 \
							-descr "Trained on \"$1\" questions, filtered dialogwise. Evaluated on \"$1\" questions filtered turnwise."

		elif [ "$4" == 'eval_on_category_dialogwise' ]; then

			echo "Evaluating on dialogs of category \"$1\". Dialogs filtered dialogwise"
			if [ "$5" != "no_wait" ]; then wait_for_approval; fi
			python evaluate.py -useGPU -evalMode ABotRank \
							-inputQues ../data/visdial_submodule/data/visdial_data_category_dialogwise/visdial_data_$1.h5 \
							-inputJson ../data/visdial_submodule/data/visdial_params_category_dialogwise/visdial_params_$1.json \
							-inputImg ../data/visdial_submodule/data/image_feats_res101_partition.h5 \
							-cocoDir ../data/visdial_submodule/data/visdial_images \
							-cocoInfo ../data/visdial_submodule/data/visdial_images/coco_info.json \
							-categoryMap ../data/visdial_submodule/data/qa_category_mapping.json  \
							-splitNames ../data/visdial_submodule/data/partition_split_names.json \
							-imgNorm 0 \
							-imgFeatureSize 2048 \
							-enableVisdom 1 \
							-visdomServerPort 8895 \
							-visdomEnv eval_$1_v3 \
							-savePath checkpoints_eval/$1_v3 \
							-saveName trained_$1_turnwise_eval_$1_dialogwise \
							-startFrom checkpoints_completed/$2 \
							-descr "Trained on \"$1\" questions, filtered turnwise. Evaluated on \"$1\" questions filtered dialogwise."


		elif [ "$4" == 'eval_all_modes' ]; then
			./scripts/eval_answerers.sh $1 $2 trained_turnwise eval_on_all_categories no_wait
			./scripts/eval_answerers.sh $1 $2 trained_turnwise eval_on_category_turnwise no_wait
			./scripts/eval_answerers.sh $1 $2 trained_turnwise eval_on_category_dialogwise no_wait

		else
			echo "Choose either \"eval_on_all_categories\", \"eval_on_category_turnwise\", or \"eval_on_category_dialogwise\" for argument 4."
			echo "Alternatively, choose \"eval_all_modes\" for argument 4 to run all these modes at once."
		fi
	fi
fi


