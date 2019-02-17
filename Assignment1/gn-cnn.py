import os
import sys
from collections import defaultdict
import time
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from collections import Counter
import pdb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--pattern", type=str, help="training patter", default="6B100d",
                    choices=['rand100', 'rand200', 'rand300', '6B100d', '6B200d', '6B300d', 'multichannel'])
parser.add_argument("-m", "--mode", type=str, help="mode of training", default="normal",
                    choices=["normal", "toy", "bylen"])
parser.add_argument("-filt", "--filter", type=int, help="filter_size", default=100)
parser.add_argument("-data", "--data", type=str, help="which dataset to use", default="all", choices=["all", "part"])
parser.add_argument("-estop", "--early_stop", type=str, help="mode of early stopping", default="immediate",
                    choices=["no", "immediate", "tolerate", "decay"])
parser.add_argument("-norm", "--weight_norm", type=float, help="norm of weight constraint", default=3)
parser.add_argument("-opt", "--opt", type=str, help="optimizer", default=["sgd", "adam"])
parser.add_argument("-wdecay", "--weight_decay", type=float, help="weight decay of optimizer", default=0)
parser.add_argument("-tr", "--ratio", type=float, help="ratio of training data", default=1)
parser.add_argument("-lyr", "--layers", type=int, help="layers of network", default=2)
parser.add_argument("-sd", "--seed", type=int, help="random seed", default=22)
parser.add_argument("-hdd", "--hidden_size", type=int, help="hidden size of cells", default=500)
parser.add_argument("-d", "--dropout", type=float, help="dropout probability", default=0.5)
parser.add_argument("-bth", "--batch", type=int, help="batch size of training", default=128)
parser.add_argument("-stop", "--stop", type=int, help="epoch of stopping tolerance", default=5)
parser.add_argument("-ep", "--epoch", type=int, help="epoch of training", default=100)
parser.add_argument("-eval", "--eval", type=int, help="evaluation iteration", default=30)
parser.add_argument("--save_model", type=bool, help="whether to save the model parameters", default=False)
args = parser.parse_args()

Tags = {}

def calc_predict_and_activations(wids, tag, words, Type):
    if len(wids) < WIN_SIZE:
        wids += [0] * (WIN_SIZE - len(wids))
    words_tensor = torch.tensor(wids).type(Type)
    scores, activations, features = model(words_tensor, return_activations=True)
    scores = scores.squeeze().cpu().data.numpy()
    print('%d ||| %s' % (tag, ' '.join(words)))
    predict = np.argmax(scores)
    print(display_activations(words, activations))
    W = model.projection_layer.weight.data.cpu().numpy()
    bias = model.projection_layer.bias.data.cpu().numpy()
    print('scores=%s, predict: %d' % (scores, predict))
    print('  bias=%s' % bias)
    contributions = W * features
    print(' very bad (%.4f): %s' % (scores[0], contributions[0]))
    print('      bad (%.4f): %s' % (scores[1], contributions[1]))
    print('  neutral (%.4f): %s' % (scores[2], contributions[2]))
    print('     good (%.4f): %s' % (scores[3], contributions[3]))
    print('very good (%.4f): %s' % (scores[4], contributions[4]))

def display_activations(words, activations):
    pad_begin = (WIN_SIZE - 1) / 2
    pad_end = WIN_SIZE - 1 - pad_begin
    words_padded = ['pad' for _ in range(int(pad_begin))] + words + ['pad' for _ in range(int(pad_end))]

    ngrams = []
    for act in activations:
        ngrams.append('[' + ', '.join(words_padded[act:act + WIN_SIZE]) + ']')

    return ngrams

class batch_loader():
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx][0], self.data[idx][2])

def collate_padding(unpadded_data):
    max_len = 60
    padded_data = [[],[],[]]
    for _, wids, tag in unpadded_data:
        if len(wids) < max_len:
            word_ids = wids + [0]*(max_len - len(wids))
        else:
            word_ids = wids[:max_len]
        padded_data[1].append(word_ids)
        padded_data[2].append(tag)
    return padded_data

class CNNclass(torch.nn.Module):
    def __init__(self, nwords, emb_size, num_filters, window_sizes, dropout, ntags, weight_norm, Type, pretrained_embedding=None):
        super(CNNclass, self).__init__()

        """ layers """
        self.embedding = torch.nn.Embedding(nwords, emb_size)
        if pretrained_embedding is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embedding).type(Type))
        else:
            # uniform initialization
            torch.nn.init.uniform_(self.embedding.weight, -0.25, 0.25)
        # Conv 1d
        self.conv_1d = torch.nn.Conv1d(in_channels=emb_size, out_channels=num_filters, kernel_size=window_sizes[0],
                                       stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv_2d = torch.nn.Conv1d(in_channels=emb_size, out_channels=num_filters, kernel_size=window_sizes[1],
                                       stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv_3d = torch.nn.Conv1d(in_channels=emb_size, out_channels=num_filters, kernel_size=window_sizes[2],
                                       stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.relu = torch.nn.ReLU()
        # Drop out layer
        self.drop_layer = torch.nn.Dropout(p=dropout)
        self.projection_layer = torch.nn.Linear(in_features=3*num_filters, out_features=ntags, bias=True)
        # Initializing the projection layer
        torch.nn.init.xavier_uniform_(self.projection_layer.weight)
        self.weight_norm = weight_norm

    def forward(self, words, return_activations=False):
        emb = self.embedding(words)                 # nwords x emb_size
        if len(emb.size()) == 3:
            batch = emb.size()[0]
            emb = emb.permute(0, 2, 1)
        else:
            batch = 1
            emb = emb.unsqueeze(0).permute(0, 2, 1)     # 1 x emb_size x nwords

        # emb of size [batch, embedding_size, sentence_length]
        # h of size [batch, filter_size, sentence_length - window_size + 1]
        h1 = self.conv_1d(emb).max(dim=2)[0]
        h2 = self.conv_2d(emb).max(dim=2)[0]
        h3 = self.conv_3d(emb).max(dim=2)[0]

        h_flat = torch.cat([h1, h2, h3], dim=1)                    # [batch, 3*filter]

        # activation operation receives size of [batch, filter_size, sentence_length - window_size + 1]
        #  activation [batch, sentence_length - window_size + 1] argmax along length of the sentence
        # the max operation reduce the filter_size dimension and select the index ones
        activations = h_flat.max(dim=1)[1]

        # Do max pooling
        h_flat = self.relu(h_flat)
        features = h_flat.squeeze(0)               # [batch, 3*filter]
        h = self.drop_layer(features)
        out = self.projection_layer(h)              # size(out) = 1 x ntags
        if return_activations:
            return out, activations.data.cpu().numpy(), features.data.cpu().numpy()
        return out


np.set_printoptions(linewidth=np.nan, threshold=np.nan)

# Functions to read in the corpus
w2i = defaultdict(lambda: len(w2i))
UNK = w2i["<unk>"]
def read_dataset(filename):
    global Tags
    with open(filename, "r") as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            words = words.split(" ")
            if Tags.get(tag, -1) < 0:
                Tags[tag] = len(Tags)

            yield (words, [w2i[x] for x in words], Tags[tag])

def load_embedding(gloveFile):
    f = open(gloveFile, 'r', encoding='utf-8')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.", len(model), " words loaded!")
    return model

def use_embedding(vocab_num, pattern):
    if pattern == 'rand100':
        pretrained_embedding = None
        EMB_SIZE = 100
    if pattern == 'rand200':
        pretrained_embedding = None
        EMB_SIZE = 200
    if pattern == 'rand300':
        pretrained_embedding = None
        EMB_SIZE = 300
    else:
        pretrained_embedding = []
        if args.pattern == '6B100d':
            # Load Glove data
            embedding = load_embedding('./embedding/glove.6B/glove.6B.100d.txt')
            EMB_SIZE = 100

        if args.pattern == '6B200d':
            embedding = load_embedding('./embedding/glove.6B/glove.6B.200d.txt')
            EMB_SIZE = 200

        if args.pattern == '6B300d':
            embedding = load_embedding('./embedding/glove.6B/glove.6B.300d.txt')
            EMB_SIZE = 300
        UNK_embedding = np.random.rand(EMB_SIZE)
        for w in i2w.values():
            try:
                pretrained_embedding.append(embedding[w])
            except KeyError:
                # pretrained_embedding.append(np.random.rand(EMB_SIZE))
                pretrained_embedding.append(UNK_embedding)

        pretrained_embedding = np.array(pretrained_embedding).reshape((vocab_num, EMB_SIZE))
    return EMB_SIZE, pretrained_embedding

def Test_output(dev):
    global Type
    for words, wids, tag in dev:
        calc_predict_and_activations(wids, tag, words, Type)
        input()
def test_set_output(model, test, index2tag, to_pad, time):
    result = []
    for words, wids, tag in test:
        if len(wids) < to_pad:
            wids += [0] * (to_pad - len(wids))
        words_tensor = torch.tensor(wids).type(Type)
        scores = model(words_tensor)
        predict = scores.argmax(dim=0).item()
        result.append(index2tag[predict] + ' ||| ' + ' '.join(words))
    try:
        filename = './test_out/{}.txt'.format(time)
        with open(filename, 'w') as fp:
            for x in result:
                fp.write(x + '\n')

        fp.close()
    except:
        pdb.set_trace()
    print('Test result save to {}'.format(filename))


def model_saver(model, PATH):
    torch.save(model.state_dict(), PATH)

def summary_saver(summary):
    summary_file = 'summary.csv'
    columns = list(summary.keys())
    if not os.path.isfile(summary_file):
        summary_data = pd.DataFrame(columns=columns)
    else:
        summary_data = pd.read_csv(summary_file)

    # result = [summary[key] for key in columns]
    summary_data = summary_data.append(summary, ignore_index=True)
    summary_data.to_csv(summary_file)
    print('Summary saved at ', summary_file)

if __name__ == "__main__":
    start = time.time()
    # Read in the data
    if args.data == "part":
        train = list(read_dataset("./data/train.txt"))
        w2i = defaultdict(lambda: UNK, w2i)
        dev = list(read_dataset("./data/dev.txt"))
    if args.data == "all":
        train = list(read_dataset("./full_data/topicclass-v1/topicclass/topicclass_train.txt"))
        # train = list(read_dataset("./full_data/topicclass-v1/topicclass/topicclass_valid.txt"))
        w2i = defaultdict(lambda: UNK, w2i)
        dev = list(read_dataset("./full_data/topicclass-v1/topicclass/topicclass_valid.txt"))
        test = list(read_dataset("./full_data/topicclass-v1/topicclass/topicclass_test.txt"))

    index2Tags = {v:k for k, v in Tags.items()}
    if args.ratio < 1:
        train = train[:int(args.ratio * len(train))]
    index2tag = {v:k for k,v in Tags.items()}
    i2w = {v: k for k, v in w2i.items()}
    nwords = len(i2w)
    numtag = {'all': 16, 'part': 5}
    ntags = numtag[args.data]
    summary = {}
    EMB_SIZE, pretrained_embedding = use_embedding(len(i2w), args.pattern)
    # Define the model
    WIN_SIZE = [3, 4, 5]
    FILTER_SIZE = args.filter
    dropout = args.dropout
    batch_size = args.batch
    NEPOCH = args.epoch
    batch_num = int(len(train) / batch_size)
    stop_warning = 0

    Type = torch.LongTensor
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if use_cuda:
        Type = torch.cuda.LongTensor
        device = 'cuda'
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.seed_all()
    else:
        device = 'cpu'

    # initialize the model
    model = CNNclass(nwords, EMB_SIZE, FILTER_SIZE, WIN_SIZE, dropout, ntags, args.weight_norm, Type, pretrained_embedding)
    criterion = torch.nn.CrossEntropyLoss()
    if args.opt == 'sgd':
        if args.weight_decay > 0:
            optimizer = optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=args.weight_decay)
        else:
            optimizer = optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


    if args.opt == 'adam':
        if args.weight_decay > 0:
            optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.Adam(model.parameters())

    if use_cuda:
        model.cuda()

    print('Begin Training, stopping after {} of decrease'.format(args.stop))
    best_model = 0
    loss = []
    train_acc = []
    test_acc = []
    for ITER in range(NEPOCH):
        # Perform training
        model.train()
        random.shuffle(train)
        train_loss = 0.0
        train_correct = 0.0
        data_count = 0
        start = time.time()
        # batches = batch_generator(data=train, length_indexing=train_length_hash, batch_size=batch_size)
        batches = DataLoader(train, batch_size=batch_size, \
                            shuffle=True, num_workers=4, collate_fn=collate_padding)
        for train_batch in batches:
            train_predict = []
            train_tags = []
            data_count += len(train_batch)

            # Perform training
            batch_ids = train_batch[1]
            batch_tags = train_batch[2]
            words_tensor = torch.tensor(batch_ids).type(Type)
            tag_tensor = torch.tensor([batch_tags]).type(Type)

            scores = model(words_tensor)  # batch * 1 * ntags
            # predicts = scores.argmax(dim=2).squeeze(1).cpu().numpy()
            predicts = scores.argmax(dim=1).cpu().numpy()
            train_predict += predicts.tolist()
            train_tags += batch_tags
            train_correct += sum(predicts == batch_tags)
            my_loss = criterion(scores, tag_tensor.squeeze(0))
            train_loss += my_loss.item()
            # Do back-prop
            optimizer.zero_grad()
            my_loss.backward()
            optimizer.step()

        loss.append(train_loss/len(train))
        train_acc.append(train_correct/len(train))
        print("iter %r: train loss/sent=%.4f, acc=%.4f, time=%.2fs" % \
              (ITER, train_loss/len(train), train_correct/len(train), time.time()-start))

        # Perform testing
        test_correct = 0.0
        model.eval()
        to_pad = 60
        for _, wids, tag in dev:
            # Padding (can be done in the conv layer as well)
            if len(wids) < to_pad:
                wids += [0] * (to_pad - len(wids))
            words_tensor = torch.tensor(wids).type(Type)
            scores = model(words_tensor)
            predict = scores.argmax(dim=0).item()
            test_correct += int(predict==tag)

        test_accuracy = test_correct/len(dev)
        if test_accuracy >= best_model:
            best_model = test_accuracy
            print("iter %r: test acc=%.4f best=%.4f" % (ITER, test_accuracy, best_model))
            test_acc.append(test_accuracy)
            if best_model >= 0.82:
                test_set_output(model, test, index2tag, 60, 'g300_f100_bat50_decay_adam_d5_{}'.format(round(best_model, 3)*1000))

        else:
            print('Decrease')
            print("iter %r: test acc=%.4f best=%.4f decay" % (ITER, test_accuracy, best_model))
            test_acc.append(test_accuracy)
            if ITER > 10:
                print('Model stopped for more than 5 epoch')
                break
            else:
                if ITER >= 5:
                    if args.early_stop == 'no':
                        break

                    if args.early_stop == 'immediate':
                        print('Model stopped')
                        break
                    if args.early_stop == 'tolerate':
                        print('STOP WARNING EPOCH ', stop_warning)
                        stop_warning += 1
                        if stop_warning > args.stop:
                            print('Model stopped')
                            break
                    if args.early_stop == 'decay':
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= 0.5
                        print('STOP WARNING EPOCH ', stop_warning)
                        stop_warning += 1
                        if stop_warning > args.stop:
                            print('Model stopped')
                            break

    if args.data == 'all':
        # Writing to summary
        summary['train_loss'] = loss
        summary['train_acc'] = train_acc
        summary['test_acc'] = test_acc
        summary['max_test_acc'] = max(test_acc)
        summary['best_epoch'] = test_acc.index(max(test_acc))

        # Writing hyperparameters
        summary['embedding'] = args.pattern
        summary['batch_method'] = args.mode
        summary['batch'] = args.batch
        summary['filter'] = args.filter
        summary['dropout'] = args.dropout
        summary['ratio'] = args.ratio
        summary['weight_norm'] = args.weight_norm

        summary['weight_decay'] = args.weight_decay
        summary['optimizer'] = args.opt
        summary['random_seed'] = args.seed
        summary_saver(summary)
        print('Data saved ...')

    pdb.set_trace()
    test_set_output(model, test, index2tag, 60, 'g100_f100_bat50_decay_adam_d5')
    Test_output(dev)
