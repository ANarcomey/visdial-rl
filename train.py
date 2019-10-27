import os
import gc
import random
import pprint
from six.moves import range
from markdown2 import markdown
from time import gmtime, strftime
from timeit import default_timer as timer
import json
import logging
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import options
from dataloader import VisDialDataset
from torch.utils.data import DataLoader
from eval_utils.rank_answerer import rankABot, rankABot_category_specific
from eval_utils.rank_questioner import rankQBot
from utils import utilities as utils
from utils.visualize import VisdomVisualize

#from pympler.tracker import SummaryTracker
#tracker = SummaryTracker()

#---------------------------------------------------------------------------------------
# Setup
#---------------------------------------------------------------------------------------

# Read the command line options
params = options.readCommandLine()

# Seed rng for reproducibility
random.seed(params['randomSeed'])
torch.manual_seed(params['randomSeed'])
if params['useGPU']:
    torch.cuda.manual_seed_all(params['randomSeed'])

# Setup dataloader
splits = ['train', 'val', 'test']
dataset = VisDialDataset(params, splits)

# Params to transfer from dataset
transfer = ['vocabSize', 'numOptions', 'numRounds']
for key in transfer:
    if hasattr(dataset, key):
        params[key] = getattr(dataset, key)

# Create save path and checkpoints folder
os.makedirs('checkpoints', exist_ok=True)
#while os.path.exists(params['savePath']):
#    params['savePath'] += '_duplicate'
if params['clobberSave'] and os.path.exists(params['savePath']):
    shutil.rmtree(params['savePath'])
os.makedirs(params['savePath'])

# Config logging
log_format = '%(levelname)-8s %(message)s'
logfile = os.path.join(params['savePath'], 'train.log')
logging.basicConfig(filename=logfile, level=logging.INFO, format=log_format)
logging.getLogger().addHandler(logging.StreamHandler())
if params['descr']:
    logging.info('='*80)
    logging.info('DESCRIPTION: '+ params['descr'])
    logging.info('='*80)

logging.info('='*80)
logging.info('PARAMS:')
logging.info(json.dumps(params))
logging.info('='*80)

# +1 - greater the better
# -1 - lower the better
metric_trends = {'r1':1, 'r5':1,'r10':1,'mean':-1,'mrr':-1,'logProbsMean':1}
metrics_by_epoch = {}

# Loading Modules
parameters = []
aBot = None
qBot = None

# Loading A-Bot
if params['trainMode'] in ['sl-abot', 'rl-full-QAf']:
    aBot, loadedParams, optim_state = utils.loadModel(params, 'abot')
    for key in loadedParams:
        params[key] = loadedParams[key]
    parameters.extend(aBot.parameters())

# Loading Q-Bot
if params['trainMode'] in ['sl-qbot', 'rl-full-QAf']:
    qBot, loadedParams, optim_state = utils.loadModel(params, 'qbot')
    for key in loadedParams:
        params[key] = loadedParams[key]

    if params['trainMode'] == 'rl-full-QAf' and params['freezeQFeatNet']:
        qBot.freezeFeatNet()
    # Filtering parameters which require a gradient update
    parameters.extend(filter(lambda p: p.requires_grad, qBot.parameters()))
    # parameters.extend(qBot.parameters())

# Setup pytorch dataloader
dataset.split = 'train'
dataloader = DataLoader(
    dataset,
    batch_size=params['batchSize'],
    shuffle=False,
    num_workers=params['numWorkers'],
    drop_last=True,
    collate_fn=dataset.collate_fn,
    pin_memory=False)

def print_sequence(seq):
    for b_idx, s in enumerate(seq):
        print("\ndialog {} of the batch:".format(b_idx))
        sentence = " ".join([dataset.ind2word[ind] for ind in s])
        print(sentence)

# Load naming of data splits
split_names = json.load(open(params['splitNames'],'r'))

# Load category specification
if params['qaCategory'] and params['categoryMap']:
    category_mapping = json.load(open(params['categoryMap'],'r'))
    train_split_name = split_names['train']
    val_split_name = split_names['val']
    category_mapping_train = category_mapping[train_split_name][params['qaCategory']]
    category_mapping_val = category_mapping[val_split_name][params['qaCategory']]

# Initializing visdom environment for plotting data
viz = VisdomVisualize(
    enable=bool(params['enableVisdom']),
    env_name=params['visdomEnv'],
    server=params['visdomServer'],
    port=params['visdomServerPort'])
pprint.pprint(params)
viz.addText(pprint.pformat(params, indent=4))

# Setup optimizer
if params['continue']:
    # Continuing from a loaded checkpoint restores the following
    startIterID = params['ckpt_iterid'] + 1  # Iteration ID
    lRate = params['ckpt_lRate']  # Learning rate
    print("Continuing training from iterId[%d]" % startIterID)
else:
    # Beginning training normally, without any checkpoint
    lRate = params['learningRate']
    startIterID = 0

optimizer = optim.Adam(parameters, lr=lRate)
if params['continue']:  # Restoring optimizer state
    print("Restoring optimizer state dict from checkpoint")
    optimizer.load_state_dict(optim_state)
runningLoss = None

mse_criterion = nn.MSELoss(reduce=False)

numIterPerEpoch = dataset.numDataPoints['train'] // params['batchSize']
print('\n%d iter per epoch.' % numIterPerEpoch)

if params['useCurriculum']:
    if params['continue']:
        rlRound = max(0, 9 - (startIterID // numIterPerEpoch))
    else:
        rlRound = params['numRounds'] - 1
else:
    rlRound = 0

#---------------------------------------------------------------------------------------
# Training
#---------------------------------------------------------------------------------------


def batch_iter(dataloader):
    for epochId in range(params['numEpochs']):
        for idx, batch in enumerate(dataloader):
            yield epochId, idx, batch


start_t = timer()
num_batches_processed = 0
num_batches_processed_epoch = 0
for epochId, idx, batch in batch_iter(dataloader):
    # Keeping track of iterId and epoch
    iterId = startIterID + idx + (epochId * numIterPerEpoch)
    epoch = iterId // numIterPerEpoch
    gc.collect()

    # Moving current batch to GPU, if available
    if dataset.useGPU:
        batch = {key: v.cuda() if hasattr(v, 'cuda') \
                                    else v for key, v in batch.items()}

    image = Variable(batch['img_feat'], requires_grad=False)
    caption = Variable(batch['cap'], requires_grad=False)
    captionLens = Variable(batch['cap_len'], requires_grad=False)
    gtQuestions = Variable(batch['ques'], requires_grad=False)
    gtQuesLens = Variable(batch['ques_len'], requires_grad=False)
    gtAnswers = Variable(batch['ans'], requires_grad=False)
    gtAnsLens = Variable(batch['ans_len'], requires_grad=False)
    options = Variable(batch['opt'], requires_grad=False)
    optionLens = Variable(batch['opt_len'], requires_grad=False)
    gtAnsId = Variable(batch['ans_id'], requires_grad=False)
    convId = Variable(batch['conv_id'], requires_grad=False)

    # Initializing optimizer and losses
    optimizer.zero_grad()
    loss = 0
    qBotLoss = 0
    aBotLoss = 0
    rlLoss = 0
    featLoss = 0
    qBotRLLoss = 0
    aBotRLLoss = 0
    predFeatures = None
    initialGuess = None
    numRounds = params['numRounds']
    # numRounds = 1 # Override for debugging lesser rounds of dialog

    # Setting training modes for both bots and observing captions, images where needed
    if aBot:
        aBot.train(), aBot.reset()
        aBot.observe(-1, image=image, caption=caption, captionLens=captionLens)
    if qBot:
        qBot.train(), qBot.reset()
        qBot.observe(-1, caption=caption, captionLens=captionLens)

    # Q-Bot image feature regression ('guessing') only occurs if Q-Bot is present
    if params['trainMode'] in ['sl-qbot', 'rl-full-QAf']:
        initialGuess = qBot.predictImage()
        prevFeatDist = mse_criterion(initialGuess, image)
        featLoss += torch.mean(prevFeatDist)
        prevFeatDist = torch.mean(prevFeatDist,1)

    # Get conversation category mapping for the batch
    if params['qaCategory'] and params['categoryMap']:
        category_mapping_conv = [category_mapping_train.get(str(batched_convId),[]) for batched_convId in convId.data]
        entire_batch_empty = True
        for category_rounds in category_mapping_conv:
            if len(category_rounds) > 0: entire_batch_empty = False
        if entire_batch_empty: continue #print("skipping batch {}".format(idx)); continue
        num_batches_processed += 1
        num_batches_processed_epoch += 1

    else:
        category_mapping_conv = None


    # Iterating over dialog rounds
    for round in range(numRounds):
        '''
        Loop over rounds of dialog. Currently three modes of training are
        supported:

            sl-abot :
                Supervised pre-training of A-Bot model using cross
                entropy loss with ground truth answers

            sl-qbot :
                Supervised pre-training of Q-Bot model using cross
                entropy loss with ground truth questions for the
                dialog model and mean squared error loss for image
                feature regression (i.e. image prediction)

            rl-full-QAf :
                RL-finetuning of A-Bot and Q-Bot in a cooperative
                setting where the common reward is the difference
                in mean squared error between the current and
                previous round of Q-Bot's image prediction.

                Annealing: In order to ease in the RL objective,
                fine-tuning starts with first N-1 rounds of SL
                objective and last round of RL objective - the
                number of RL rounds are increased by 1 after
                every epoch until only RL objective is used for
                all rounds of dialog.

        '''
        # Tracking components which require a forward pass
        # A-Bot dialog model


        forwardABot = (params['trainMode'] == 'sl-abot'
                       or (params['trainMode'] == 'rl-full-QAf'
                           and round < rlRound))
        # Q-Bot dialog model
        forwardQBot = (params['trainMode'] == 'sl-qbot'
                       or (params['trainMode'] == 'rl-full-QAf'
                           and round < rlRound))
        # Q-Bot feature regression network
        forwardFeatNet = (forwardQBot or params['trainMode'] == 'rl-full-QAf')

        # Answerer Forward Pass
        if forwardABot:
            # Observe Ground Truth (GT) question
            aBot.observe(
                round,
                ques=gtQuestions[:, round],
                quesLens=gtQuesLens[:, round])
            # Observe GT answer for teacher forcing
            aBot.observe(
                round,
                ans=gtAnswers[:, round],
                ansLens=gtAnsLens[:, round])
            ansLogProbs = aBot.forward()
            # Cross Entropy (CE) Loss for Ground Truth Answers
            aBotLoss += utils.maskedNll_byCategory(ansLogProbs,
                                        gtAnswers[:, round].contiguous(),
                                        category_mapping_conv, round)

        # Questioner Forward Pass (dialog model)
        if forwardQBot:
            # Observe GT question for teacher forcing
            qBot.observe(
                round,
                ques=gtQuestions[:, round],
                quesLens=gtQuesLens[:, round])
            quesLogProbs = qBot.forward()
            # Cross Entropy (CE) Loss for Ground Truth Questions
            qBotLoss += utils.maskedNll_byCategory(quesLogProbs,
                                        gtQuestions[:, round].contiguous(),
                                        category_mapping_conv, round)
            # Observe GT answer for updating dialog history
            qBot.observe(
                round,
                ans=gtAnswers[:, round],
                ansLens=gtAnsLens[:, round])

        # In order to stay true to the original implementation, the feature
        # regression network makes predictions before dialog begins and for
        # the first 9 rounds of dialog. This can be set to 10 if needed.
        MAX_FEAT_ROUNDS = 9

        # Questioner feature regression network forward pass
        if forwardFeatNet and round < MAX_FEAT_ROUNDS:
            # Make an image prediction after each round
            predFeatures = qBot.predictImage()
            featDist = mse_criterion(predFeatures, image)
            featDist = torch.mean(featDist)
            featLoss += featDist

        # A-Bot and Q-Bot interacting in RL rounds
        if params['trainMode'] == 'rl-full-QAf' and round >= rlRound:
            # Run one round of conversation
            questions, quesLens = qBot.forwardDecode(inference='sample')
            qBot.observe(round, ques=questions, quesLens=quesLens)
            aBot.observe(round, ques=questions, quesLens=quesLens)
            answers, ansLens = aBot.forwardDecode(inference='sample')
            aBot.observe(round, ans=answers, ansLens=ansLens)
            qBot.observe(round, ans=answers, ansLens=ansLens)

            # Q-Bot makes a guess at the end of each round
            predFeatures = qBot.predictImage()

            # Computing reward based on Q-Bot's predicted image
            featDist = mse_criterion(predFeatures, image)
            featDist = torch.mean(featDist,1)

            reward = prevFeatDist.detach() - featDist
            prevFeatDist = featDist

            qBotRLLoss = qBot.reinforce(reward)
            if params['rlAbotReward']:
                aBotRLLoss = aBot.reinforce(reward)
            rlLoss += torch.mean(aBotRLLoss)
            rlLoss += torch.mean(qBotRLLoss)

    # Loss coefficients
    rlCoeff = 1
    rlLoss = rlLoss * rlCoeff
    featLoss = featLoss * params['featLossCoeff']
    # Averaging over rounds
    qBotLoss = (params['CELossCoeff'] * qBotLoss) / numRounds
    aBotLoss = (params['CELossCoeff'] * aBotLoss) / numRounds
    featLoss = featLoss / numRounds  #/ (numRounds+1)
    rlLoss = rlLoss / numRounds
    # Total loss
    loss = qBotLoss + aBotLoss + rlLoss + featLoss
    loss.backward()
    optimizer.step()

    # Tracking a running average of loss
    if runningLoss is None:
        runningLoss = loss.data[0]
    else:
        runningLoss = 0.95 * runningLoss + 0.05 * loss.data[0]

    # Decay learning rate
    if lRate > params['minLRate']:
        for gId, group in enumerate(optimizer.param_groups):
            optimizer.param_groups[gId]['lr'] *= params['lrDecayRate']
        lRate *= params['lrDecayRate']
        if iterId % 10 == 0:  # Plot learning rate till saturation
            viz.linePlot(iterId, lRate, 'learning rate', 'learning rate')

    # RL Annealing: Every epoch after the first, decrease rlRound
    if iterId % numIterPerEpoch == 0 and iterId > 0:
        if params['trainMode'] == 'rl-full-QAf':
            rlRound = max(0, rlRound - 1)
            print('Using rl starting at round {}'.format(rlRound))

    # Print every now and then
    if iterId % 10 == 0:
        end_t = timer()  # Keeping track of iteration(s) time
        curEpoch = float(iterId) / numIterPerEpoch
        timeStamp = strftime('%a %d %b %y %X', gmtime())
        printFormat = '[%s][Ep: %.2f][Iter: %d][Time: %5.2fs][Loss: %.3g]'
        printFormat += '[lr: %.3g]'
        printInfo = [
            timeStamp, curEpoch, iterId, end_t - start_t, loss.data[0], lRate
        ]
        start_t = end_t
        print(printFormat % tuple(printInfo))

        # Update line plots
        if isinstance(aBotLoss, Variable):
            viz.linePlot(iterId, aBotLoss.data[0], 'aBotLoss', 'train CE')
        if isinstance(qBotLoss, Variable):
            viz.linePlot(iterId, qBotLoss.data[0], 'qBotLoss', 'train CE')
        if isinstance(rlLoss, Variable):
            viz.linePlot(iterId, rlLoss.data[0], 'rlLoss', 'train')
        if isinstance(featLoss, Variable):
            viz.linePlot(iterId, featLoss.data[0], 'featLoss',
                         'train FeatureRegressionLoss')
        viz.linePlot(iterId, loss.data[0], 'loss', 'train loss')
        viz.linePlot(iterId, runningLoss, 'loss', 'running train loss')


    # Evaluate every epoch
    if iterId % (numIterPerEpoch // 1) == 0 and iterId != 0:
        # Keeping track of epochID
        curEpoch = float(iterId) / numIterPerEpoch
        epochId = (1.0 * iterId / numIterPerEpoch) + 1

        # Set eval mode
        if aBot:
            aBot.eval()
        if qBot:
            qBot.eval()

        if params['enableVisdom']:
            # Printing visdom environment name in terminal
            print("Currently on visdom env [%s]" % (params['visdomEnv']))

        # Mapping iteration count to epoch count
        viz.linePlot(iterId, epochId, 'iter x epoch', 'epochs')
        logging.info("======================================================")
        logging.info("Epoch {}, iter {}".format(epochId, iterId))
        logging.info("Cummulative Num batches processed: {}".format(num_batches_processed))
        logging.info("Num batches processed this epoch: {}".format(num_batches_processed_epoch))
        num_batches_processed_epoch = 0


        logging.info('Performing validation...')
        if aBot and 'ques' in batch:
            logging.info("aBot Validation:")

            # NOTE: A-Bot validation is slow, so adjust exampleLimit as needed
            
            '''
            import pdb;pdb.set_trace()
            rankMetrics = rankABot(
                aBot,
                dataset,
                'val',
                scoringFunction=utils.maskedNll,
                #exampleLimit=25 * params['batchSize'])
                #exampleLimit=None)
                exampleLimit=None)

            rankMetrics2 = rankABot_category_specific_batchless(
                aBot,
                dataset,
                'val',
                params['categoryMap'],
                params['qaCategory'],
                scoringFunction=utils.maskedNll,
                exampleLimit=params['batchSize'])

            rankMetrics3 = rankABot_category_specific(
                aBot,
                dataset,
                'val',
                params['categoryMap'],
                params['qaCategory'],
                scoringFunction=utils.maskedNll,
                exampleLimit=None)

            rankMetrics4 = rankABot_category_specific_v2(
                aBot,
                dataset,
                'val',
                params['categoryMap'],
                params['qaCategory'],
                scoringFunction=utils.maskedNll_byCategory,
                exampleLimit=params['batchSize'])
            '''


            if params['qaCategory'] and params['categoryMap']:
                #import pdb;pdb.set_trace()
                logging.info("Performing category-filtered validation on val data.")
                rankMetrics_category_filter = rankABot_category_specific(
                    aBot,
                    dataset,
                    'val',
                    params['qaCategory'],
                    category_mapping_val,
                    scoringFunction=utils.maskedNll,
                    exampleLimit=None)
                    #exampleLimit=params['batchSize'])
                rankMetrics = rankMetrics_category_filter

                '''rankMetrics_no_filter = rankABot(
                    aBot,
                    dataset,
                    'val',
                    scoringFunction=utils.maskedNll,
                    exampleLimit=None)
                    #exampleLimit=params['batchSize'])

                rankMetrics_test = rankABot(
                    aBot,
                    dataset,
                    'test',
                    scoringFunction=utils.maskedNll,
                    exampleLimit=None)
                    #exampleLimit=params['batchSize'])'''

            else:
                logging.info("Performing validation on val data, no category filtering at this stage.")
                rankMetrics_no_filter = rankABot(
                    aBot,
                    dataset,
                    'val',
                    scoringFunction=utils.maskedNll,
                    exampleLimit=None)
                    #exampleLimit=params['batchSize'])
                rankMetrics = rankMetrics_no_filter

            for metric, value in rankMetrics.items():
                viz.linePlot(
                    epochId, value, 'val - aBot', metric, xlabel='Epochs')

            metrics_by_epoch[epochId] = rankMetrics
            best_epoch_by_metric = {}
            metrics_by_name = {}
            for metric_name, trend in metric_trends.items():
                metric_values = [(values[metric_name], metric_epochId) for metric_epochId, values in metrics_by_epoch.items()]
                if trend == -1:
                    best_value, best_value_epoch = min(metric_values)
                elif trend == 1:
                    best_value, best_value_epoch = max(metric_values)
                else:
                    raise ValueError("Invalid trend of {} for metric {}".format(trend, metric_name))
                metrics_by_name[metric_name] = [{'epoch':metric_epochId, 'value':value} for value,metric_epochId in metric_values]
                best_epoch_by_metric[metric_name] = {'epoch':best_value_epoch, 'value':best_value}

            json.dump(metrics_by_epoch, open(os.path.join(params['savePath'], 'metrics_by_epoch.json'),'w'))  
            logging.info("validation metrics for this epoch: \n{}".format(rankMetrics))
            logging.info("validation metric history arranged by metric name: \n{}".format(metrics_by_name))
            logging.info("validation metric best epoch by metric: \n{}".format(best_epoch_by_metric))


            if 'logProbsMean' in rankMetrics:
                logProbsMean = params['CELossCoeff'] * rankMetrics[
                    'logProbsMean']
                viz.linePlot(iterId, logProbsMean, 'aBotLoss', 'val CE')

                if params['trainMode'] == 'sl-abot':
                    valLoss = logProbsMean
                    viz.linePlot(iterId, valLoss, 'loss', 'val loss')

        if qBot:
            print("qBot Validation:")
            rankMetrics, roundMetrics = rankQBot(qBot, dataset, 'val')

            for metric, value in rankMetrics.items():
                viz.linePlot(
                    epochId, value, 'val - qBot', metric, xlabel='Epochs')

            viz.linePlot(iterId, epochId, 'iter x epoch', 'epochs')

            if 'logProbsMean' in rankMetrics:
                logProbsMean = params['CELossCoeff'] * rankMetrics[
                    'logProbsMean']
                viz.linePlot(iterId, logProbsMean, 'qBotLoss', 'val CE')

            if 'featLossMean' in rankMetrics:
                featLossMean = params['featLossCoeff'] * (
                    rankMetrics['featLossMean'])
                viz.linePlot(iterId, featLossMean, 'featLoss',
                             'val FeatureRegressionLoss')

            if 'logProbsMean' in rankMetrics and 'featLossMean' in rankMetrics:
                if params['trainMode'] == 'sl-qbot':
                    valLoss = logProbsMean + featLossMean
                    viz.linePlot(iterId, valLoss, 'loss', 'val loss')

    # Save the model after every epoch
        params['ckpt_iterid'] = iterId
        params['ckpt_lRate'] = lRate

        if aBot:
            saveFile = os.path.join(params['savePath'],
                                    'abot_ep_%d.vd' % curEpoch)
            print('Saving model: ' + saveFile)
            utils.saveModel(aBot, optimizer, saveFile, params)
        if qBot:
            saveFile = os.path.join(params['savePath'],
                                    'qbot_ep_%d.vd' % curEpoch)
            print('Saving model: ' + saveFile)
            utils.saveModel(qBot, optimizer, saveFile, params)
