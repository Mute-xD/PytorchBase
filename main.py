import torch  # torch must be imported at first
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import utils
import model
import dataset
import tqdm

# TODO PLAN
"""
"""


class Overlap:
    def __init__(self):
        self.config = utils.Config('./config.yml')
        self.device = None
        self.summary = {}
        self.dumpPath = None
        self.sysConfig()
        self.setSummary()
        self.pipeRaw = self.loadDataset()
        self.pipeLen = self.pipeRaw['train'].__len__()
        self.pipe = None
        self.pipeIter()
        self.gen = model.Generator(self.config)
        self.dis = model.Discriminator(self.config)

        if self.config.GPU == -1 and self.config.CUDA:
            print('Using MultiGPU')
            self.gen = nn.parallel.DataParallel(self.gen).to(self.device)

        else:
            self.gen = self.gen.to(self.device)

        self.optGen = torch.optim.Adam(self.gen.parameters(), lr=eval(self.config.LR), betas=self.config.BETA)
        self.optDis = torch.optim.Adam(self.dis.parameters(), lr=eval(self.config.LR), betas=self.config.BETA)

    def build(self, img, summary=False):
        self.gen.train()

        fake = self.buildGenerator(img)
        feed = [img, fake]
        l1 = F.l1_loss(img, fake)
        disOut = self.buildDiscriminator(feed)
        genLoss = l1 + disOut['wGAN-g']
        disLoss = disOut['wGAN-d']
        self.optGen.zero_grad()
        genLoss.backward()
        self.optGen.step()

        self.optDis.zero_grad()
        disLoss.backward()
        self.optDis.step()

        if summary:
            self.summary['writer'].add_scalar('d-loss', disLoss.item(), global_step=self.summary['epochs'])
            self.summary['writer'].add_scalar('g-loss', genLoss.item(), global_step=self.summary['epochs'])
            summaryImg = torch.cat((img[0], fake[0]), dim=-1)

            self.summary['writer'].add_image('train/img-fake',
                                             summaryImg, global_step=self.summary['epochs'])

    def buildDiscriminator(self, feed):
        img, fake = feed

        '''
        W-GAN loss
        '''
        neg = self.dis(fake)
        pos = self.dis(img)
        dLoss = torch.mean(neg - pos)
        gLoss = -torch.mean(neg).detach()
        return {'wGAN-d': dLoss, 'wGAN-g': gLoss}

    def buildGenerator(self, img):
        fake = self.gen(img)
        return fake

    def train(self, epochs, batch):
        img = next(self.pipe['train'])
        img = img.to(self.device)

        self.summary['epochs'] = epochs
        self.summary['batch'] = batch

        if epochs % self.config.SUMMARY_PER_EPOCHS == 0 and self.summary['flag']:
            self.build(img, summary=True)
            self.summary['flag'] = False
            for valIndex in tqdm.tqdm(range(0, 10, self.config.BATCH_SIZE), desc='Validating...'):
                val = next(self.pipe['val'])
                val = val.to(self.device)
                self.buildVal(val, valIndex)
            self.test()
        else:
            self.build(img, summary=False)

    def trainManager(self, epochsLimit, batchLimit=None, epochs=0):
        print('Target Epochs: ', self.config.EPOCHS_LIMIT)
        if batchLimit is None:
            batchLimit = self.pipeLen
        print('Target Batch: ', batchLimit)
        print('Summary in Every', self.config.SUMMARY_PER_EPOCHS, 'Epochs')
        print('Save Model in Every', self.config.CHECKPOINT, 'Epochs')
        while epochs < epochsLimit:
            for batch in tqdm.tqdm(range(batchLimit), desc=''.join(('Training epochs -> ', str(epochs + 1)))):
                self.train(epochs, batch)
            self.pipeIter()
            epochs += 1
            if self.config.SUMMARY:
                self.summary['flag'] = True
            if epochs % self.config.CHECKPOINT == 0:
                print('Checkpoint:', epochs)
                self.modelSaver(epochs)
        self.modelSaver('final')
        print('\033[32mDone!\033[0m')

    @torch.no_grad()
    def buildVal(self, val, valIndex):
        self.gen.eval()
        self.dis.eval()
        fake = self.buildGenerator(val)
        feed = [val, fake]
        l1 = F.l1_loss(fake, val)
        disOut = self.buildDiscriminator(feed)

        disLoss = disOut['wGAN-d']
        genLoss = l1 + disOut['wGAN-g']

        if valIndex == 0:
            self.summary['writer'].add_scalar('val/d-loss', disLoss.item(), global_step=self.summary['epochs'])
            self.summary['writer'].add_scalar('val/g-loss', genLoss.item(), global_step=self.summary['epochs'])

        for b in range(self.config.BATCH_SIZE):
            summaryImg = torch.cat((val[b], fake[b]), dim=-1)
            self.summary['writer'].add_image('val/' + str(valIndex + b),
                                             summaryImg, global_step=self.summary['epochs'])

    @torch.no_grad()
    def buildTest(self, real):
        self.gen.eval()

        fake = self.buildGenerator(real, )
        return fake

    def saveOutputImg(self, fakeOverlap):
        import matplotlib.pyplot as plt
        import os
        # epochs = self.summary['epochs']
        batch = self.summary['batch']
        imgSavePath = ''.join((self.dumpPath, 'imgSave/'))
        fileNameA = ''.join((imgSavePath, str(batch * 2), '.png'))
        fileNameB = ''.join((imgSavePath, str(batch * 2 + 1), '.png'))
        fakeOverlap = fakeOverlap.to('cpu').detach()
        if not os.path.exists(imgSavePath):
            os.makedirs(imgSavePath)
        plt.imsave(fileNameA, fakeOverlap[0].permute(1, 2, 0).numpy())
        plt.imsave(fileNameB, fakeOverlap[1].permute(1, 2, 0).numpy())

    def loadDataset(self):
        fileList = np.load(self.config.TRAIN_FILELIST).tolist()
        fileListVal = np.load(self.config.VAL_FILELIST).tolist()
        fileListTest = np.load(self.config.TEST_FILELIST).tolist()
        if self.config.RAM_DATASET:
            print('\033[32mUsing RAM Dataset\033[0m')
            datasetTrain = dataset.RAMDataset(self.config, fileList)
            datasetVal = dataset.RAMDataset(self.config, fileListVal)
            datasetTest = dataset.RAMDataset(self.config, fileListTest)
        else:
            print('\033[32mUsing Normal Dataset\033[0m')
            datasetTrain = dataset.Dataset(self.config, fileList)
            datasetVal = dataset.Dataset(self.config, fileListVal)
            datasetTest = dataset.Dataset(self.config, fileListTest)
        pipeTrain = torch.utils.data.DataLoader(datasetTrain,
                                                batch_size=self.config.BATCH_SIZE, drop_last=True, shuffle=True)
        pipeVal = torch.utils.data.DataLoader(datasetVal, batch_size=self.config.BATCH_SIZE, drop_last=True)
        pipeTest = torch.utils.data.DataLoader(datasetTest, batch_size=3)

        return {'train': pipeTrain, 'val': pipeVal, 'test': pipeTest}

    def sysConfig(self):
        import datetime

        if not self.config.RANDOM:
            torch.manual_seed(1234)
            np.random.seed(1234)
        # torch.autograd.set_detect_anomaly(True)  # debug

        print('\033[32mCUDA available: \033[0m', torch.cuda.is_available(),
              '\n\033[32mUsing CUDA: \033[0m', self.config.CUDA)
        if self.config.CUDA:
            self.device = 'cuda'
            if self.config.GPU != -1:
                torch.cuda.set_device(self.config.GPU)
                print('device: ', torch.cuda.get_device_name(self.config.GPU), 'id:', torch.cuda.current_device())
                print('CUDA:', torch.version.cuda)
            else:
                print('Using All GPUs')

            if self.config.CUDNN:
                import torch.backends.cudnn as cudnn
                print('\033[32mUsing CUDNN\033[0m')
                cudnn.enabled = True
                cudnn.benchmark = True
        else:
            self.device = 'cpu'

        # get timestamp
        current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        self.dumpPath = ''.join((
            './', 'runs', '/', self.config.FRIENDLY_NAME, '_', current_time, '/'))
        utils.configDumper('./config.yml', self.dumpPath)
        print('Dump Path:', self.dumpPath)

    def modelViewer(self, model_, input_size=None):
        import torchsummary
        if input_size is None:
            input_size = [3]
            input_size.extend(self.config.IMG_SHAPE)
        torchsummary.summary(model_, tuple(input_size), device=self.device, batch_size=self.config.BATCH_SIZE)

    def pipeIter(self):
        """
        set or reset pipeline iterator
        need to call when a epochs is finished
        :return: None
        """
        self.pipe = {key: iter(value) for key, value in self.pipeRaw.items()}

    def setSummary(self):
        from torch.utils.tensorboard import SummaryWriter
        if self.config.SUMMARY:
            print('\033[32mSummary will be saved\033[0m')
            self.summary['writer'] = SummaryWriter(log_dir=self.dumpPath)

        self.summary['flag'] = False

    def modelSaver(self, epochs):
        model_ = {'gen': self.gen.state_dict(),
                  'dis': self.dis.state_dict(),
                  'epochs': epochs}
        path = ''.join((self.dumpPath, 'model_saved_', str(epochs), '.pkl'))
        with open(path, 'wb') as fh:
            torch.save(model_, fh, pickle_protocol=-1)
        print('\nModel saved at: ', path)

    def modelLoader(self, path):
        with open(path, 'rb') as fh:
            modelSaved = torch.load(fh)
        self.gen.load_state_dict(modelSaved['gen'])
        self.dis.load_state_dict(modelSaved['dis'])
        # epochs can be loaded like
        return modelSaved['epochs']

    def test(self):
        testingBatch = self.pipe['test'].__len__()
        for i in range(testingBatch):
            test = next(self.pipe['test'])
            test = test.unsqueeze(0).to(self.device)
            fake = self.buildTest(test)
            save = torch.cat((test, fake), dim=-1)[0]
            self.summary['writer'].add_image('test/' + str(i),
                                             save, global_step=self.summary['epochs'])

    def run(self):
        if self.config.RESUME:
            epoch = self.modelLoader(self.config.DUMPER_PATH)
            self.trainManager(self.config.EPOCHS_LIMIT, epoch)
        else:
            self.trainManager(self.config.EPOCHS_LIMIT)


if __name__ == '__main__':
    overlap = Overlap()
    overlap.run()
