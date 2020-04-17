import os
import librosa
import time
import warnings
import numpy as np
import torch 
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from dataset import logger
from radam import RAdam

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TasNET_trainer(object):
    def __init__(self,
                 TasNET,
                 batch_size,
                 checkpoint="checkpoint",
                 log_folder="./log",
                 optimizer='radam',
                 lr=1e-5,
                 momentum=0.9,
                 weight_decay=0,
                 num_epoches=20,
                 clip_norm=False,
                 sr=8000,
                 cudnnBenchmark=True):
        
        self.TasNET = TasNET

        self.log_folder = log_folder
        self.writer = SummaryWriter(log_folder)
        self.all_log = 'all_log.log' #all log filename
        self.log('Progress Log save path: '+log_folder)

        self.log("TasNET:\n{}".format(self.TasNET))
        if type(lr) is str:
            lr = float(lr)
            logger.info("Transfrom lr from str to float => {}".format(lr))
        
        self.log('Batch size used: '+str(batch_size))

        if optimizer == 'radam':
            self.log('Using RAdam optimizer')
            self.optimizer = RAdam(self.TasNET.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            self.log('Using Adam optimizer (default)')
            self.optimizer = torch.optim.Adam(
                self.TasNET.parameters(),
                lr=lr,
                weight_decay=weight_decay)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                            'min', factor=0.5, patience=3,verbose=True)
        self.TasNET.to(device)

        self.checkpoint = checkpoint
        self.log('Model save path: '+checkpoint)
        
        self.num_epoches = num_epoches
        self.clip_norm = clip_norm
        self.sr = sr

        if self.clip_norm:
            self.log("Clip gradient by 2-norm {}".format(clip_norm))

        if not os.path.exists(self.checkpoint):
            os.makedirs(checkpoint)
            
        torch.backends.cudnn.benchmark=cudnnBenchmark
        self.log('cudnn benchmark status: '+str(torch.backends.cudnn.benchmark))


    def SISNR(self, output, target):
        #output:(128,4000)
        batchsize = np.shape(output)[0]
        target = target.view(batchsize,-1)
        output = output - torch.mean(output,1,keepdim=True)
        target = target - torch.mean(target,1,keepdim=True)

        s_shat = torch.sum(output*target,1,keepdim=True)
        s_2 = torch.sum(target**2,1,keepdim=True)
        s_target = (s_shat / s_2) * target   #(128,4000)

        e_noise = output - s_target    

        return 10*torch.log10(torch.sum(e_noise**2,1,keepdim=True)\
                    /torch.sum(s_target**2,1,keepdim=True))        #(128,1)


    def loss(self,output1,output2,target1,target2):
    	#PIT loss
        loss1 = self.SISNR(output1,target1)+self.SISNR(output2,target2)
        loss2 = self.SISNR(output1,target2)+self.SISNR(output2,target1)
        min = torch.min(loss1, loss2)   #(128,1)
        return torch.mean(min)        #scale

    def train(self, dataloader, epoch):
        self.TasNET.train()
        self.log("Training...")
        tot_loss = 0
        tot_batch = len(dataloader)
        batch_indx = (epoch-1)*tot_batch

        currProcess = 0
        fivePercentProgress = tot_batch//20

        for mix_speech, speech1, speech2 in dataloader:
            self.optimizer.zero_grad()

            if torch.cuda.is_available():
                mix_speech= mix_speech.cuda()
                speech1 = speech1.cuda()
                speech2 = speech2.cuda()

            mix_speech = Variable(mix_speech)
            speech1 = Variable(speech1)
            speech2 = Variable(speech2)

            output1, output2 = self.TasNET(mix_speech)
            cur_loss = self.loss(output1,output2,speech1,speech2)
            tot_loss += cur_loss.item()
            
            #write summary
            batch_indx += 1
            self.writer.add_scalar('train_loss', cur_loss, batch_indx)
            cur_loss.backward()
            if self.clip_norm:
                nn.utils.clip_grad_norm_(self.TasNET.parameters(),
                                         self.clip_norm)
            self.optimizer.step()
            currProcess+=1
            if currProcess % fivePercentProgress == 0:
                self.log('batch {}: {:.2f}% progress ({}/{})| LR: {}'.format(batch_indx, currProcess*100/tot_batch, currProcess, tot_batch, str(self.get_curr_lr())))
                
        return tot_loss / tot_batch, tot_batch

    def validate(self, dataloader, epoch):
        """one epoch"""
        self.TasNET.eval()
        self.log("Evaluating...")
        tot_loss = 0
        tot_batch = len(dataloader)
        batch_indx = (epoch-1)*tot_batch

        currProcess = 0
        fivePercentProgress = tot_batch//20
        #print(tot_batch)

        with torch.no_grad():
            for mix_speech,speech1,speech2 in dataloader:
                if torch.cuda.is_available():
                    mix_speech = mix_speech.cuda()
                    speech1 = speech1.cuda()
                    speech2 = speech2.cuda()

                mix_speech = Variable(mix_speech)
                speech1 = Variable(speech1)
                speech2 = Variable(speech2)

                output1, output2 = self.TasNET(mix_speech)
                cur_loss = self.loss(output1,output2,speech1,speech2)
                tot_loss += cur_loss.item()
                #write summary
                batch_indx += 1
                currProcess += 1
                if currProcess % fivePercentProgress == 0:
                    self.log('batch {}: {:.2f}% progress ({}/{})| LR: {}'.format(batch_indx, currProcess*100/tot_batch, currProcess, tot_batch, str(self.get_curr_lr())))
                self.writer.add_scalar('dev_loss', cur_loss, batch_indx)
        return tot_loss / tot_batch, tot_batch

    def run(self, train_set, dev_set):
        init_loss, _ = self.validate(dev_set,1)
        self.log("Start training for {} epoches".format(self.num_epoches))
        self.log("Epoch {:2d}: dev loss ={:.4e}".format(0, init_loss))
        torch.save(self.TasNET.state_dict(), os.path.join(self.checkpoint, 'TasNET_0.pkl'))
        for epoch in range(1, self.num_epoches+1):
            train_start = time.time()
            train_loss, train_num_batch = self.train(train_set, epoch)
            valid_start = time.time()
            valid_loss, valid_num_batch = self.validate(dev_set, epoch)
            valid_end = time.time()
            self.scheduler.step(valid_loss)
            self.log(
                "Epoch {:2d}: train loss = {:.4e}({:.2f}s/{:d}) |"
                " dev loss= {:.4e}({:.2f}s/{:d})".format(
                    epoch, train_loss, valid_start - train_start,
                    train_num_batch, valid_loss, valid_end - valid_start,
                    valid_num_batch))
            save_path = os.path.join(
                self.checkpoint, "TasNET_{:d}_trainloss_{:.4e}_valloss_{:.4e}.pkl".format(
                    epoch, train_loss, valid_loss))
            torch.save(self.TasNET.state_dict(), save_path)
        self.log("Training for {} epoches done!".format(self.num_epoches))
    
    def rerun(self, train_set, dev_set, model_path, epoch_done):
        self.TasNET.load_state_dict(torch.load(model_path))
        # init_loss, _ = self.validate(dev_set,epoch_done)
        # logger.info("Start training for {} epoches".format(self.num_epoches))
        # logger.info("Epoch {:2d}: dev loss ={:.4e}".format(0, init_loss))
        # torch.save(self.TasNET.state_dict(), os.path.join(self.checkpoint, 'TasNET_0.pkl'))
        for epoch in range(epoch_done+1, self.num_epoches+1):
            train_start = time.time()
            train_loss, train_num_batch = self.train(train_set,epoch)
            valid_start = time.time()
            valid_loss, valid_num_batch = self.validate(dev_set,epoch)
            valid_end = time.time()
            self.scheduler.step(valid_loss)
            self.log(
                "Epoch {:2d}: train loss = {:.4e}({:.2f}s/{:d}) |"
                " dev loss= {:.4e}({:.2f}s/{:d})".format(
                    epoch, train_loss, valid_start - train_start,
                    train_num_batch, valid_loss, valid_end - valid_start,
                    valid_num_batch))
            save_path = os.path.join(
                self.checkpoint, "TasNET_{:d}_trainloss_{:.4e}_valloss_{:.4e}.pkl".format(
                    epoch, train_loss, valid_loss))
            torch.save(self.TasNET.state_dict(), save_path)
        self.log("Training for {} epoches done!".format(self.num_epoches))

    def get_curr_lr(self):
        for i, param_group in enumerate(self.optimizer.param_groups):
            curr_lr = float(param_group['lr'])
            return curr_lr
    
    def log(self, log_data):
        logger.info(log_data)
        try:
            f = open(self.log_folder+'/'+self.all_log,'a+')
            f.write(log_data+'\n')
            f.close()
        except:
            logger.info('failed to save last log')
