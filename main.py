'''
Parts of this code were incorporated from the following github repositories:
1. parksunwoo/show_attend_and_tell_pytorch
Link: https://github.com/parksunwoo/show_attend_and_tell_pytorch/blob/master/prepro.py

2. sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
Link: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning

This script has the Encoder and Decoder models and training/validation scripts. 
Edit the parameters sections of this file to specify which models to load/run
''' 

# coding: utf-8
import pickle
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from data_loader import get_loader
from nltk.translate.bleu_score import corpus_bleu
from processData import Vocabulary
from tqdm import tqdm
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import skimage.transform
from imageio import imread
# from scipy.misc import imresize
from skimage.transform import resize as imresize
from PIL import Image
import matplotlib.image as mpimg
from torchtext.vocab import Vectors, GloVe
from scipy import misc
from pytorch_pretrained_bert import BertTokenizer, BertModel
import imageio
import json

###################
# START Parameters
###################

# hyperparams
grad_clip = 5.
num_epochs = 4
batch_size = 1
decoder_lr = 0.0004

# if both are false them model = baseline

# glove_model = [False, False]
# bert_model = [False, True]

from_checkpoint = True
train_model = False
valid_model = True
metric = False


###################
# END Parameters
###################

# loss
class loss_obj(object):
    def __init__(self):
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load pre-trained model (weights)
BertModel = BertModel.from_pretrained('bert-base-uncased').to(device)
BertModel.eval()


# Load GloVe
glove_vectors = pickle.load(open('glove.6B/glove_words.pkl', 'rb'))
glove_vectors = torch.tensor(glove_vectors)

#####################
# Encoder RASNET CNN
#####################
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        resnet = models.resnet101(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))

    def forward(self, images):
        out = self.adaptive_pool(self.resnet(images))
        # batch_size, img size, imgs size, 2048
        out = out.permute(0, 2, 3, 1)
        return out

####################
# Attention Decoder
####################
class Decoder(nn.Module):

    def __init__(self, vocab_size, use_glove, use_bert):
        super(Decoder, self).__init__()
        self.encoder_dim = 2048
        self.attention_dim = 512
        self.use_bert = use_bert

        if use_glove:
            self.embed_dim = 300
        elif use_bert:
            self.embed_dim = 768
        else:
            self.embed_dim = 512

        self.decoder_dim = 512
        self.vocab_size = vocab_size
        self.dropout = 0.5
        
        # soft attention
        self.enc_att = nn.Linear(2048, 512)
        self.dec_att = nn.Linear(512, 512)
        self.att = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        # decoder layers
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(self.embed_dim + self.encoder_dim, self.decoder_dim, bias=True)
        self.h_lin = nn.Linear(self.encoder_dim, self.decoder_dim)
        self.c_lin = nn.Linear(self.encoder_dim, self.decoder_dim)
        self.f_beta = nn.Linear(self.decoder_dim, self.encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(self.decoder_dim, self.vocab_size)

        # init variables
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)
        
        if not use_bert:
            self.embedding = nn.Embedding(vocab_size, self.embed_dim)
            self.embedding.weight.data.uniform_(-0.1, 0.1)

            # load Glove embeddings
            if use_glove:
                self.embedding.weight = nn.Parameter(glove_vectors)

            # always fine-tune embeddings (even with GloVe)
            for p in self.embedding.parameters():
                p.requires_grad = True

    def forward(self, encoder_out):    
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size
        k = 10 #beam_size
        encoded_captions = torch.LongTensor([[1]]*k).to(device)

        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)

        #Testing

        # load bert or regular embeddings
        def dec_embedding(encoded_captions, num_pixels):
            max_cap_len = 1 
            if not self.use_bert:
                embeddings = self.embedding(encoded_captions)
            elif self.use_bert:
                embeddings = []
                sentences = []
                for cap_idx in  encoded_captions:
                    cap_idx = cap_idx.tolist()
                
                    # padd caption to correct size
                    while len(cap_idx) < max_cap_len:
                        cap_idx.append(PAD)
                    
                    cap = ' '.join([vocab.idx2word[word_idx] for word_idx in cap_idx])
                    sentences.append(cap)
                    #embeddings = torch.tensor(bert_Transformer.encode(sentences))
                    cap = u'[CLS] '+cap
                
                    tokenized_cap = tokenizer.tokenize(cap)                
                    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_cap)
                    tokens_tensor = torch.tensor([indexed_tokens]).to(device)

                    with torch.no_grad():
                        encoded_layers, cls_head = BertModel(tokens_tensor)
                    bert_embedding = encoded_layers[11].squeeze(0)
                
                    split_cap = cap.split()
                    tokens_embedding = []
                    j = 0

                    for full_token in split_cap:
                        curr_token = ''
                        x = 0
                        for i,_ in enumerate(tokenized_cap[1:]): # disregard CLS
                            token = tokenized_cap[i+j]
                            piece_embedding = bert_embedding[i+j]
                        
                            # full token
                            if token == full_token and curr_token == '' :
                                tokens_embedding.append(piece_embedding)
                                j += 1
                                break
                            else: # partial token
                                x += 1
                            
                                if curr_token == '':
                                    tokens_embedding.append(piece_embedding)
                                    curr_token += token.replace('#', '')
                                else:
                                    tokens_embedding[-1] = torch.add(tokens_embedding[-1], piece_embedding)
                                    curr_token += token.replace('#', '')
                                
                                    if curr_token == full_token: # end of partial
                                        j += x
                                        break

                    cap_embedding = torch.stack(tokens_embedding)
                    embeddings.append(cap_embedding)
 
                embeddings = torch.stack(embeddings)
                embeddings = embeddings[:,1,:]
            return embeddings

        # init hidden state
        avg_enc_out = encoder_out.mean(dim=1)
        h = self.h_lin(avg_enc_out)
        c = self.c_lin(avg_enc_out)


        # Tensor to store top k sequences' scores
        k_prev_words = encoded_captions
        seqs = k_prev_words
        top_k_scores = torch.zeros(k, 1).to(device)

        seqs_alpha = torch.ones(k, 1, encoder_dim, encoder_dim).to(device)

        complete_seqs = list()
        complete_seqs_scores = list()

        #for t in range(max(dec_len)):
        step = 1
        while True:
            embeddings = dec_embedding(k_prev_words, self.encoder_dim).squeeze(1)
            # soft-attention
            enc_att = self.enc_att(encoder_out)
            dec_att = self.dec_att(h)
            att = self.att(self.relu(enc_att + dec_att.unsqueeze(1))).squeeze(2)
            alpha = self.softmax(att)
            attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        
            gate = self.sigmoid(self.f_beta(h))
            attention_weighted_encoding = gate * attention_weighted_encoding

            cat_val = torch.cat([embeddings.double(), attention_weighted_encoding.double()], dim=1)
            
            h, c = self.decode_step(cat_val.float(),(h.float(), c.float()))
            # preds = self.fc(self.dropout(h))
            preds = self.fc(h)
            # Testing
            preds = F.log_softmax(preds, dim=1)
            preds = top_k_scores.expand_as(preds) + preds

            if step == 1:
                top_k_scores, top_k_words = preds[0].topk(k, 0, True, True)
            else:
                top_k_scores, top_k_words = preds.view(-1).topk(k, 0, True, True)

            prev_word_inds = (top_k_words / len(vocab)).long()
            next_word_inds = (top_k_words % len(vocab)).long()

            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)

            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != END]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)

            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1

            # predictions[:batch_size_t, t, :] = preds
            # alphas[:batch_size_t, t, :] = alpha

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]

        # preds, sorted capts, dec lens, attention wieghts
        return seq

# vocab indices
PAD = 0
START = 1
END = 2
UNK = 3

# Load vocabulary
with open('data/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

# load data
train_loader = get_loader('train', vocab, batch_size)
val_loader = get_loader('val', vocab, batch_size)

#############
# Init model
#############

criterion = nn.CrossEntropyLoss().to(device)
if from_checkpoint:

    bert_encoder = Encoder().to(device)
    baseline_encoder = Encoder().to(device)
    glove_encoder = Encoder().to(device)
    bert_decoder = Decoder(vocab_size=len(vocab), use_glove=False, use_bert=True).to(device)
    baseline_decoder = Decoder(vocab_size=len(vocab), use_glove=False, use_bert=False).to(device)
    glove_decoder = Decoder(vocab_size=len(vocab), use_glove=True, use_bert=False).to(device)

    if torch.cuda.is_available():
        if bert_model:
            print('Pre-Trained BERT Model')
            encoder_checkpoint = torch.load('./checkpoints/encoder_bert')
            decoder_checkpoint = torch.load('./checkpoints/decoder_bert')
        elif glove_model:
            print('Pre-Trained GloVe Model')
            encoder_checkpoint = torch.load('./checkpoints/encoder_glove')
            decoder_checkpoint = torch.load('./checkpoints/decoder_glove')
        else:
            print('Pre-Trained Baseline Model')
            encoder_checkpoint = torch.load('./checkpoints/TrainingCheckpoints/Baseline/encoder_epoch4.zip')
            decoder_checkpoint = torch.load('./checkpoints/TrainingCheckpoints/Baseline/decoder_epoch4.zip')
    else:
        #if bert_model:
        print('Pre-Trained BERT Model')
        bert_encoder_checkpoint = torch.load('./checkpoints/TrainingCheckpoints/BERT/encoder_epoch4.zip', map_location='cpu')
        bert_decoder_checkpoint = torch.load('./checkpoints/TrainingCheckpoints/BERT/decoder_epoch4.zip', map_location='cpu')
        #elif glove_model:
        print('Pre-Trained GloVe Model')
        glove_encoder_checkpoint = torch.load('./checkpoints/TrainingCheckpoints/GloVe/encoder_epoch4.zip', map_location='cpu')
        glove_decoder_checkpoint = torch.load('./checkpoints/TrainingCheckpoints/GloVe/decoder_epoch4.zip', map_location='cpu')
        #else:
        print('Pre-Trained Baseline Model')
        baseline_encoder_checkpoint = torch.load('./checkpoints/TrainingCheckpoints/Baseline/encoder_epoch4.zip', map_location='cpu')
        baseline_decoder_checkpoint = torch.load('./checkpoints/TrainingCheckpoints/Baseline/decoder_epoch4.zip', map_location='cpu')

    bert_encoder.load_state_dict(bert_encoder_checkpoint['model_state_dict'])
    baseline_encoder.load_state_dict(baseline_encoder_checkpoint['model_state_dict'])
    #glove_encoder.load_state_dict(glove_encoder_checkpoint['model_state_dict'])
    bert_decoder_optimizer = torch.optim.Adam(params=bert_decoder.parameters(),lr=decoder_lr)
    baseline_decoder_optimizer = torch.optim.Adam(params=baseline_decoder.parameters(),lr=decoder_lr)
    #glove_decoder_optimizer = torch.optim.Adam(params=glove_decoder.parameters(),lr=decoder_lr)
    bert_decoder.load_state_dict(bert_decoder_checkpoint['model_state_dict'])
    baseline_decoder.load_state_dict(baseline_decoder_checkpoint['model_state_dict'])
    #glove_decoder.load_state_dict(glove_decoder_checkpoint['model_state_dict'])
    bert_decoder_optimizer.load_state_dict(bert_decoder_checkpoint['optimizer_state_dict'])
    baseline_decoder_optimizer.load_state_dict(baseline_decoder_checkpoint['optimizer_state_dict'])
    #glove_decoder_optimizer.load_state_dict(glove_decoder_checkpoint['optimizer_state_dict'])
else:
    encoder = Encoder().to(device)
    decoder = Decoder(vocab_size=len(vocab),use_glove=glove_model, use_bert=bert_model).to(device)
    decoder_optimizer = torch.optim.Adam(params=decoder.parameters(),lr=decoder_lr)

###############
# Train model
###############

def train():
    print("Started training...")
    for epoch in tqdm(range(num_epochs)):
        decoder.train()
        encoder.train()

        losses = loss_obj()
        num_batches = len(train_loader)

        for i, (imgs, caps, caplens) in enumerate(tqdm(train_loader)):

            imgs = encoder(imgs.to(device))
            caps = caps.to(device)

            scores, caps_sorted, decode_lengths, alphas = decoder(imgs, caps, caplens)
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]

            targets = caps_sorted[:, 1:]
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

            loss = criterion(scores, targets).to(device)

            loss += ((1. - alphas.sum(dim=1)) ** 2).mean()

            decoder_optimizer.zero_grad()
            loss.backward()

            # grad_clip decoder
            for group in decoder_optimizer.param_groups:
                for param in group['params']:
                    if param.grad is not None:
                        param.grad.data.clamp_(-grad_clip, grad_clip)

            decoder_optimizer.step()

            losses.update(loss.item(), sum(decode_lengths))

            # save model each 100 batches
            if i%5000==0 and i!=0:
                print('epoch '+str(epoch+1)+'/4 ,Batch '+str(i)+'/'+str(num_batches)+' loss:'+str(losses.avg))
                
                 # adjust learning rate (create condition for this)
                for param_group in decoder_optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.8

                print('saving model...')

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': decoder.state_dict(),
                    'optimizer_state_dict': decoder_optimizer.state_dict(),
                    'loss': loss,
                    }, './checkpoints/decoder_mid')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': encoder.state_dict(),
                    'loss': loss,
                    }, './checkpoints/encode_mid')

                print('model saved')

        torch.save({
            'epoch': epoch,
            'model_state_dict': decoder.state_dict(),
            'optimizer_state_dict': decoder_optimizer.state_dict(),
            'loss': loss,
            }, './checkpoints/decoder_epoch'+str(epoch+1))

        torch.save({
            'epoch': epoch,
            'model_state_dict': encoder.state_dict(),
            'loss': loss,
            }, './checkpoints/encoder_epoch'+str(epoch+1))
        print('epoch checkpoint saved')

    print("Completed training...")  

#################
# Validate model
#################

def print_sample_from_json(hypotheses, references, test_references, k, avg_loss):
    bleu_1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    bleu_2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
    bleu_3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
    bleu_4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

    print("Validation loss: "+str(avg_loss))
    print("BLEU-1: "+str(bleu_1))
    print("BLEU-2: "+str(bleu_2))
    print("BLEU-3: "+str(bleu_3))
    print("BLEU-4: "+str(bleu_4))


def print_sample(hypotheses, references, test_references,imgs, alphas, k, show_att, losses):
    bleu_1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    bleu_2 = corpus_bleu(references, hypotheses, weights=(0, 1, 0, 0))
    bleu_3 = corpus_bleu(references, hypotheses, weights=(0, 0, 1, 0))
    bleu_4 = corpus_bleu(references, hypotheses, weights=(0, 0, 0, 1))

    print("Validation loss: "+str(losses.avg))
    print("BLEU-1: "+str(bleu_1))
    print("BLEU-2: "+str(bleu_2))
    print("BLEU-3: "+str(bleu_3))
    print("BLEU-4: "+str(bleu_4))

    img_dim = 336 # 14*24
    
    hyp_sentence = []
    for word_idx in hypotheses[k]:
        hyp_sentence.append(vocab.idx2word[word_idx])
    
    ref_sentence = []
    for word_idx in test_references[k]:
        ref_sentence.append(vocab.idx2word[word_idx])

    print('Hypotheses: '+" ".join(hyp_sentence))
    print('References: '+" ".join(ref_sentence))
        
    img = imgs[0][k] 
    imageio.imwrite('img.jpg', img)
  
    if show_att:
        image = Image.open('img.jpg')
        image = image.resize([img_dim, img_dim], Image.LANCZOS)
        for t in range(len(hyp_sentence)):

            plt.subplot(np.ceil(len(hyp_sentence) / 5.), 5, t + 1)

            plt.text(0, 1, '%s' % (hyp_secapsntence[t]), color='black', backgroundcolor='white', fontsize=12)
            plt.imshow(image)
            current_alpha = alphas[0][t, :].detach().numpy()
            alpha = skimage.transform.resize(current_alpha, [img_dim, img_dim])
            if t == 0:
                plt.imshow(alpha, alpha=0)
            else:
                plt.imshow(alpha, alpha=0.7)
            plt.axis('off')
    else:
        img = imageio.imread('img.jpg')
        plt.imshow(img)
        plt.axis('off')
        plt.show()


def validate(model_name, encoder, decoder):

    references = [] 
    test_references = []
    hypotheses = [] 
    all_imgs = []
    all_alphas = []

    print("Started validation...")
    decoder.eval()
    encoder.eval()

    losses = loss_obj()


    num_batches = len(val_loader)
    # Batches
    data = {}
    init_cap = None
    init_caplens = None
    #Definitely not the most efficient way to do it but take 32 random captions to initialize with
    for i, (imgs, caps, caplens, img_ids) in enumerate(val_loader):
        init_cap = caps
        init_caplens = caplens
        break


    for i, (imgs, caps, caplens, img_ids) in enumerate(tqdm(val_loader)):
        imgs_jpg = imgs.numpy() 
        imgs_jpg = np.swapaxes(np.swapaxes(imgs_jpg, 1, 3), 1, 2)
        
        # Forward prop.
        imgs = encoder(imgs.to(device))
        caps = caps.to(device)

        # Just to determine max caption length
        max_len = max(caplens)
        # word2idx('<start>') = 1
        k_prev_words = torch.ones(max_len, max_len).long()
        k_prev_words.to(device)

        #Testing:
        pred = decoder(imgs)

        #Batch size is 1 so:
        img_id = img_ids[0]
        caption_list_idxs = val_loader.dataset.coco.getAnnIds(imgIds=img_id)
        caption_list = [val_loader.dataset.coco.anns[ann]['caption'] for ann in caption_list_idxs]
        pred = [w for w in pred if w not in [PAD, START, END]]
        data[img_id] = dict()
        data[img_id]['reference'] = caption_list
        data[img_id]['hypothesis'] = ' '.join([vocab.idx2word[idx] for idx in pred])
        if i == 2:
            break


    with open(f'val_output/{model_name}.json', 'w') as outfile:
        json.dump(data, outfile)

    print("Completed validation...")

######################
# Run training/validation
######################

if train_model:
    train()

if valid_model:
    # print('Validating Baseline')
    # validate('Baseline', baseline_encoder, baseline_decoder)
    print('Validating GloVe')
    validate('GloVe', glove_encoder, glove_decoder)
    #print('Validating BERT')
    #validate('BERT_test', bert_encoder, bert_decoder)

if metric:
    with open('val_output/BERT.json') as json_file:
        data = json.load(json_file)
    hypotheses = data['hypotheses']
    references = data['references']
    test_references = data['test_references']
    avg_loss = data['avg_loss']

    convert_json(hypotheses, test_references, outfile='val_output/BERT_v2.json')
    
    with open('val_output/Baseline.json') as json_file:
        data = json.load(json_file)
    hypotheses = data['hypotheses']
    references = data['references']
    test_references = data['test_references']
    avg_loss = data['avg_loss']

    convert_json(hypotheses, test_references, outfile='val_output/Baselien_v2.json')
    
   
    #print_sample_from_json(hypotheses, references, test_references, 1, avg_loss)

