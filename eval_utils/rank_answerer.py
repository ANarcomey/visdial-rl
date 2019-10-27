import sys
import json
import h5py
import numpy as np
from timeit import default_timer as timer

import torch
from torch.autograd import Variable

import options
import visdial.metrics as metrics
from utils import utilities as utils
from dataloader import VisDialDataset
from torch.utils.data import DataLoader

from sklearn.metrics.pairwise import pairwise_distances

from six.moves import range


def rankOptions(options, gtOptions, scores):
    '''Rank a batch of examples against a list of options.'''
    numOptions = options.size(1)
    # Compute score of GT options in 'scores'
    gtScores = scores.gather(1, gtOptions.unsqueeze(1))
    # Sort all predicted scores
    sortedScore, _ = torch.sort(scores, 1)
    # In sorted scores, count how many are greater than the GT score
    ranks = torch.sum(sortedScore.gt(gtScores).float(), 1)
    return ranks + 1

""" #same outcome as rankABot_category_specific()
def rankABot_category_specific_batchless(aBot, dataset, split, categoryMappingFilename, category, scoringFunction, exampleLimit=None):
    '''
        Evaluate A-Bot performance on ranking answer option when it is
        shown ground truth image features, captions and questions.

        Arguments:
            aBot    : A-Bot
            dataset : VisDialDataset instance
            split   : Dataset split, can be 'val' or 'test'

            scoringFunction : A function which computes negative log
                              likelihood of a sequence (answer) given log
                              probabilities under an RNN model. Currently
                              utils.maskedNll is the only such function used.
            exampleLimit    : Maximum number of data points to use from
                              the dataset split. If None, all data points.
    '''
    batchSize = 1
    numRounds = dataset.numRounds
    if exampleLimit is None:
        numExamples = dataset.numDataPoints[split]
    else:
        numExamples = exampleLimit

    numBatches = (numExamples - 1) // batchSize + 1

    # Load category specification
    category_mapping = json.load(open(categoryMappingFilename,'r'))
    category_mapping_split = category_mapping[split][category]
    skipped_batches = []

    original_split = dataset.split
    dataset.split = split
    dataloader = DataLoader(
        dataset,
        batch_size=batchSize,
        shuffle=False,
        num_workers=1,
        collate_fn=dataset.collate_fn)

    totalLoss, totalTokens = 0, 0
    ranks = []
    logProbsAll = [[] for _ in range(numRounds)]
    start_t = timer()
    import pdb;pdb.set_trace()
    for idx, batch in enumerate(dataloader):
        print("idx = ", idx)
        if idx == numBatches:
            break

        if dataset.useGPU:
            batch = {
                key: v.cuda() if hasattr(v, 'cuda') else v
                for key, v in batch.items()
            }
        else:
            batch = {
                key: v.contiguous() if hasattr(v, 'cuda') else v
                for key, v in batch.items()
            }

        image = Variable(batch['img_feat'], volatile=True)
        caption = Variable(batch['cap'], volatile=True)
        captionLens = Variable(batch['cap_len'], volatile=True)
        questions = Variable(batch['ques'], volatile=True)
        quesLens = Variable(batch['ques_len'], volatile=True)
        answers = Variable(batch['ans'], volatile=True)
        ansLens = Variable(batch['ans_len'], volatile=True)
        options = Variable(batch['opt'], volatile=True)
        optionLens = Variable(batch['opt_len'], volatile=True)
        correctOptionInds = Variable(batch['ans_id'], volatile=True)
        convId = Variable(batch['conv_id'], volatile=False)

        # Get conversation category mapping for the batch
        category_mapping_conv = [category_mapping_split.get(str(batched_convId),[]) for batched_convId in convId.data]
        entire_batch_empty = True
        for category_rounds in category_mapping_conv:
            if len(category_rounds) > 0: entire_batch_empty = False
        if entire_batch_empty: skipped_batches.append(idx); continue

        aBot.reset()
        aBot.observe(-1, image=image, caption=caption, captionLens=captionLens)
        for round in range(numRounds):
            print("r = ", round)
            aBot.observe(
                round,
                ques=questions[:, round],
                quesLens=quesLens[:, round],
                ans=answers[:, round],
                ansLens=ansLens[:, round])

            logProbs = aBot.evalOptions(options[:, round],
                                        optionLens[:, round], scoringFunction) #batch x 100 options
            logProbsCurrent = aBot.forward() #batch x max answer length x vocab size

            if round in category_mapping_conv[0]:
                
                logProbsAll[round].append(
                    scoringFunction(logProbsCurrent,
                                    answers[:, round].contiguous())) 
                batchRanks = rankOptions(options[:, round],
                                         correctOptionInds[:, round], logProbs) #batch,
                ranks.append(batchRanks)

        end_t = timer()
        delta_t = " Rate: %5.2fs" % (end_t - start_t)
        start_t = end_t
        progressString = "\r[Abot] Evaluating split '%s' [%d/%d]\t" + delta_t
        sys.stdout.write(progressString % (split, idx + 1, numBatches))
        sys.stdout.flush()
    sys.stdout.write("\n")
    dataloader = None
    print("Sleeping for 3 seconds to let dataloader subprocesses exit...")
    ranks = torch.cat(ranks, 0) #list of num batches*num_rounds, each item is batchsize tensor --> flatten
    rankMetrics = metrics.computeMetrics(ranks.cpu())

    logProbsAll = [torch.cat(lprobs, 0).mean() for lprobs in logProbsAll] #list<round>:list<104 batches in dataset>:float
    roundwiseLogProbs = torch.cat(logProbsAll, 0).data.cpu().numpy() #num_rounds,
    logProbsMean = roundwiseLogProbs.mean() #float
    rankMetrics['logProbsMean'] = 1.*logProbsMean

    dataset.split = original_split
    return rankMetrics
"""

 # evaluates on all dialog turns, masks out scores to 0 for out-of-category
"""
def rankABot_category_specific_v2(aBot, dataset, split, categoryMappingFilename, category, scoringFunction, exampleLimit=None):
    '''
        Evaluate A-Bot performance on ranking answer option when it is
        shown ground truth image features, captions and questions.

        Arguments:
            aBot    : A-Bot
            dataset : VisDialDataset instance
            split   : Dataset split, can be 'val' or 'test'

            scoringFunction : A function which computes negative log
                              likelihood of a sequence (answer) given log
                              probabilities under an RNN model. Currently
                              utils.maskedNll is the only such function used.
            exampleLimit    : Maximum number of data points to use from
                              the dataset split. If None, all data points.
    '''
    batchSize = dataset.batchSize
    numRounds = dataset.numRounds
    if exampleLimit is None:
        numExamples = dataset.numDataPoints[split]
    else:
        numExamples = exampleLimit

    numBatches = (numExamples - 1) // batchSize + 1

    # Load category specification
    category_mapping = json.load(open(categoryMappingFilename,'r'))
    category_mapping_split = category_mapping[split][category]
    skipped_batches = []

    original_split = dataset.split
    dataset.split = split
    dataloader = DataLoader(
        dataset,
        batch_size=batchSize,
        shuffle=False,
        num_workers=1,
        collate_fn=dataset.collate_fn)

    totalLoss, totalTokens = 0, 0
    ranks = []
    logProbsAll = [[] for _ in range(numRounds)]
    start_t = timer()
    for idx, batch in enumerate(dataloader):
        #print("idx = ", idx)
        if idx == numBatches:
            break

        if dataset.useGPU:
            batch = {
                key: v.cuda() if hasattr(v, 'cuda') else v
                for key, v in batch.items()
            }
        else:
            batch = {
                key: v.contiguous() if hasattr(v, 'cuda') else v
                for key, v in batch.items()
            }

        image = Variable(batch['img_feat'], volatile=True)
        caption = Variable(batch['cap'], volatile=True)
        captionLens = Variable(batch['cap_len'], volatile=True)
        questions = Variable(batch['ques'], volatile=True)
        quesLens = Variable(batch['ques_len'], volatile=True)
        answers = Variable(batch['ans'], volatile=True)
        ansLens = Variable(batch['ans_len'], volatile=True)
        options = Variable(batch['opt'], volatile=True)
        optionLens = Variable(batch['opt_len'], volatile=True)
        correctOptionInds = Variable(batch['ans_id'], volatile=True)
        convId = Variable(batch['conv_id'], volatile=False)

        # Get conversation category mapping for the batch
        category_mapping_conv = [category_mapping_split.get(str(batched_convId),[]) for batched_convId in convId.data]
        entire_batch_empty = True
        for category_rounds in category_mapping_conv:
            if len(category_rounds) > 0: entire_batch_empty = False
        if entire_batch_empty: skipped_batches.append(idx); continue

        aBot.reset()
        aBot.observe(-1, image=image, caption=caption, captionLens=captionLens)
        for round in range(numRounds):
            #print("r = ", round)
            aBot.observe(
                round,
                ques=questions[:, round],
                quesLens=quesLens[:, round],
                ans=answers[:, round],
                ansLens=ansLens[:, round])

            logProbs = aBot.evalOptions(options[:, round],
                                        optionLens[:, round], utils.maskedNll_byCategory, 
                                        category_mapping_conv=category_mapping_conv,
                                        round=round) #batch x 100 options
            #category mask nll means log probs for filtered batch/round pairs are masked out

            logProbsCurrent = aBot.forward() #batch x max answer length x vocab siz

            logProbsAll[round].append(
                    utils.maskedNll_byCategory(logProbsCurrent,
                                answers[:, round].contiguous(), category_mapping_conv, round))

            batchRanks = rankOptions(options[:, round],
                                     correctOptionInds[:, round], logProbs) #batch,
            ranks.append(batchRanks)

        end_t = timer()
        delta_t = " Rate: %5.2fs" % (end_t - start_t)
        start_t = end_t
        progressString = "\r[Abot] Evaluating split '%s' [%d/%d]\t" + delta_t
        sys.stdout.write(progressString % (split, idx + 1, numBatches))
        sys.stdout.flush()
    sys.stdout.write("\n")
    dataloader = None
    print("Sleeping for 3 seconds to let dataloader subprocesses exit...")
    ranks = torch.cat(ranks, 0) #list of num batches*num_rounds, each item is batchsize tensor --> flatten
    rankMetrics = metrics.computeMetrics(ranks.cpu())

    logProbsAll = [torch.cat(lprobs, 0).mean() for lprobs in logProbsAll] #list<round>:list<104 batches in dataset>:float
    roundwiseLogProbs = torch.cat(logProbsAll, 0).data.cpu().numpy() #num_rounds,
    logProbsMean = roundwiseLogProbs.mean() #float
    rankMetrics['logProbsMean'] = 1.*logProbsMean

    dataset.split = original_split
    return rankMetrics
"""



def rankABot_category_specific(aBot, dataset, split, category, categoryFiltering, scoringFunction, exampleLimit=None):
    '''
        Evaluate A-Bot performance on ranking answer option when it is
        shown ground truth image features, captions and questions.

        Arguments:
            aBot    : A-Bot
            dataset : VisDialDataset instance
            split   : Dataset split, can be 'val' or 'test'

            scoringFunction : A function which computes negative log
                              likelihood of a sequence (answer) given log
                              probabilities under an RNN model. Currently
                              utils.maskedNll is the only such function used.
            exampleLimit    : Maximum number of data points to use from
                              the dataset split. If None, all data points.
    '''
    batchSize = dataset.batchSize
    numRounds = dataset.numRounds
    if exampleLimit is None:
        numExamples = dataset.numDataPoints[split]
    else:
        numExamples = exampleLimit

    numBatches = (numExamples - 1) // batchSize + 1

    skipped_batches = []

    original_split = dataset.split
    dataset.split = split
    dataloader = DataLoader(
        dataset,
        batch_size=batchSize,
        shuffle=True,
        num_workers=1,
        collate_fn=dataset.collate_fn)

    totalLoss, totalTokens = 0, 0
    ranks = []
    logProbsAll = [[] for _ in range(numRounds)]
    start_t = timer()
    for idx, batch in enumerate(dataloader):
        if idx == numBatches:
            break

        if dataset.useGPU:
            batch = {
                key: v.cuda() if hasattr(v, 'cuda') else v
                for key, v in batch.items()
            }
        else:
            batch = {
                key: v.contiguous() if hasattr(v, 'cuda') else v
                for key, v in batch.items()
            }

        image = Variable(batch['img_feat'], volatile=True)
        caption = Variable(batch['cap'], volatile=True)
        captionLens = Variable(batch['cap_len'], volatile=True)
        questions = Variable(batch['ques'], volatile=True)
        quesLens = Variable(batch['ques_len'], volatile=True)
        answers = Variable(batch['ans'], volatile=True)
        ansLens = Variable(batch['ans_len'], volatile=True)
        options = Variable(batch['opt'], volatile=True)
        optionLens = Variable(batch['opt_len'], volatile=True)
        correctOptionInds = Variable(batch['ans_id'], volatile=True)
        convId = Variable(batch['conv_id'], volatile=False)

        # Get conversation category mapping for the batch
        category_mapping_conv = [categoryFiltering.get(str(batched_convId),[]) for batched_convId in convId.data]
        entire_batch_empty = True
        for category_rounds in category_mapping_conv:
            if len(category_rounds) > 0: entire_batch_empty = False
        if entire_batch_empty: skipped_batches.append(idx); continue

        aBot.reset()
        aBot.observe(-1, image=image, caption=caption, captionLens=captionLens)
        for round in range(numRounds):
            aBot.observe(
                round,
                ques=questions[:, round],
                quesLens=quesLens[:, round],
                ans=answers[:, round],
                ansLens=ansLens[:, round])

            logProbs = aBot.evalOptions(options[:, round],
                                        optionLens[:, round], scoringFunction) #batch x 100 options
            logProbsCurrent = aBot.forward() #batch x max answer length x vocab size

            for bidx in range(len(convId)):
                if round in category_mapping_conv[bidx]:
                
                    logProbsAll[round].append(
                        scoringFunction(logProbsCurrent[bidx].unsqueeze(0),
                                        answers[bidx, round].unsqueeze(0).contiguous())) 
                    batchRanks = rankOptions(options[bidx, round].unsqueeze(0),
                                             correctOptionInds[bidx, round], logProbs[bidx].unsqueeze(0)) #batch,
                    ranks.append(batchRanks)

        end_t = timer()
        delta_t = " Rate: %5.2fs" % (end_t - start_t)
        start_t = end_t
        progressString = "\r[Abot] Evaluating split '%s' [%d/%d]\t" + delta_t
        sys.stdout.write(progressString % (split, idx + 1, numBatches))
        sys.stdout.flush()
    sys.stdout.write("\n")
    dataloader = None
    print("Sleeping for 3 seconds to let dataloader subprocesses exit...")
    ranks = torch.cat(ranks, 0) #list of num batches*num_rounds, each item is batchsize tensor --> flatten
    rankMetrics = metrics.computeMetrics(ranks.cpu())

    logProbsAll = [torch.cat(lprobs, 0).mean() for lprobs in logProbsAll] #list<round>:list<104 batches in dataset>:float
    roundwiseLogProbs = torch.cat(logProbsAll, 0).data.cpu().numpy() #num_rounds,
    logProbsMean = roundwiseLogProbs.mean() #float
    rankMetrics['logProbsMean'] = 1.*logProbsMean

    dataset.split = original_split
    return rankMetrics



def rankABot(aBot, dataset, split, scoringFunction, exampleLimit=None):
    '''
        Evaluate A-Bot performance on ranking answer option when it is
        shown ground truth image features, captions and questions.

        Arguments:
            aBot    : A-Bot
            dataset : VisDialDataset instance
            split   : Dataset split, can be 'val' or 'test'

            scoringFunction : A function which computes negative log
                              likelihood of a sequence (answer) given log
                              probabilities under an RNN model. Currently
                              utils.maskedNll is the only such function used.
            exampleLimit    : Maximum number of data points to use from
                              the dataset split. If None, all data points.
    '''
    batchSize = dataset.batchSize
    numRounds = dataset.numRounds
    if exampleLimit is None:
        numExamples = dataset.numDataPoints[split]
    else:
        numExamples = exampleLimit

    numBatches = (numExamples - 1) // batchSize + 1

    original_split = dataset.split
    dataset.split = split
    dataloader = DataLoader(
        dataset,
        batch_size=batchSize,
        shuffle=True,
        num_workers=1,
        collate_fn=dataset.collate_fn)

    totalLoss, totalTokens = 0, 0
    ranks = []
    logProbsAll = [[] for _ in range(numRounds)]
    start_t = timer()
    for idx, batch in enumerate(dataloader):
        if idx == numBatches:
            break

        if dataset.useGPU:
            batch = {
                key: v.cuda() if hasattr(v, 'cuda') else v
                for key, v in batch.items()
            }
        else:
            batch = {
                key: v.contiguous() if hasattr(v, 'cuda') else v
                for key, v in batch.items()
            }

        image = Variable(batch['img_feat'], volatile=True)
        caption = Variable(batch['cap'], volatile=True)
        captionLens = Variable(batch['cap_len'], volatile=True)
        questions = Variable(batch['ques'], volatile=True)
        quesLens = Variable(batch['ques_len'], volatile=True)
        answers = Variable(batch['ans'], volatile=True)
        ansLens = Variable(batch['ans_len'], volatile=True)
        options = Variable(batch['opt'], volatile=True)
        optionLens = Variable(batch['opt_len'], volatile=True)
        correctOptionInds = Variable(batch['ans_id'], volatile=True)
        aBot.reset()
        aBot.observe(-1, image=image, caption=caption, captionLens=captionLens)
        for round in range(numRounds):
            aBot.observe(
                round,
                ques=questions[:, round],
                quesLens=quesLens[:, round],
                ans=answers[:, round],
                ansLens=ansLens[:, round])
            logProbs = aBot.evalOptions(options[:, round],
                                        optionLens[:, round], scoringFunction) #batch x 100 options
            logProbsCurrent = aBot.forward() #batch x max answer length x vocab size
            logProbsAll[round].append(
                scoringFunction(logProbsCurrent,
                                answers[:, round].contiguous())) 
            batchRanks = rankOptions(options[:, round],
                                     correctOptionInds[:, round], logProbs) #batch,
            ranks.append(batchRanks)

        end_t = timer()
        delta_t = " Rate: %5.2fs" % (end_t - start_t)
        start_t = end_t
        progressString = "\r[Abot] Evaluating split '%s' [%d/%d]\t" + delta_t
        sys.stdout.write(progressString % (split, idx + 1, numBatches))
        sys.stdout.flush()
    sys.stdout.write("\n")
    dataloader = None
    print("Sleeping for 3 seconds to let dataloader subprocesses exit...")
    ranks = torch.cat(ranks, 0) #list of num batches*num_rounds, each item is batchsize tensor --> flatten
    rankMetrics = metrics.computeMetrics(ranks.cpu())

    logProbsAll = [torch.cat(lprobs, 0).mean() for lprobs in logProbsAll] #list<round>:list<104 batches in dataset>:float
    roundwiseLogProbs = torch.cat(logProbsAll, 0).data.cpu().numpy() #num_rounds,
    logProbsMean = roundwiseLogProbs.mean() #float
    rankMetrics['logProbsMean'] = 1.*logProbsMean

    dataset.split = original_split
    return rankMetrics
