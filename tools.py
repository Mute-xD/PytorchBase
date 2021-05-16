class PartImage2Box:
    def __init__(self):
        self.rawNeighbour = './raw/neighbour/'
        self.rawTarget = './raw/target/'
        self.part = './part/'
        self.partNeighbour = './part/neighbour/'
        self.partTarget = './part/target/'
        self.partVal = './part/val/'
        self.partValNeighbour = self.partVal + 'neighbour/'
        self.partValTarget = self.partVal + 'target/'
        self.real = './part/test/'
        self.realOrigin = './test/'

    def clean(self):
        import os

        def deleteFolder(folderPath):
            pos = folderPath.rfind("/")
            if pos > 0:
                folderPath = folderPath + '/'
            try:
                childList = os.listdir(folderPath)
            except Exception as e:
                return e
            if childList is None:
                print('Folder is empty')
                return
            for child in childList:
                isFile = child.rfind('.')
                if isFile > 0:
                    os.remove(folderPath + child)
                else:
                    deleteFolder(folderPath + child)
            os.rmdir(folderPath)
            return

        deleteFolder('./part/')
        os.makedirs(self.partTarget)
        os.makedirs(self.partNeighbour)
        os.makedirs(self.partValTarget)
        os.makedirs(self.partValNeighbour)

    def cut(self, batch, shape, isVal=False, form='.tif'):
        import os
        import random
        from PIL import Image
        import tqdm
        neighbourPathList = [self.rawNeighbour + i for i in os.listdir(self.rawNeighbour)]
        targetPathList = [self.rawTarget + i for i in os.listdir(self.rawTarget)]
        if len(neighbourPathList) != len(targetPathList):
            raise ResourceWarning('Check if image missed')
        picListSize = len(neighbourPathList)
        if isVal:
            targetPath = self.partValTarget
            neighbourPath = self.partValNeighbour
        else:
            targetPath = self.partTarget
            neighbourPath = self.partNeighbour
        for i in tqdm.tqdm(range(batch)):
            picIndex = random.randint(0, picListSize - 1)
            with Image.open(targetPathList[picIndex]) as img:
                imgSize = img.size
                randX = random.randint(0, imgSize[0] - shape[0])
                randY = random.randint(0, imgSize[1] - shape[1])
                box = (randX, randY, randX + shape[0], randY + shape[1])
                img = img.crop(box)
                img.save(targetPath + str(i) + form)
            with Image.open(neighbourPathList[picIndex]) as img:
                img = img.crop(box)
                img.save(neighbourPath + str(i) + form)

    def createFileList(self, val=False):
        """
        Read the folder, create a file list, save by a .npy file, auto shuffled
        neighbour will be related with fListA


        :parameter val: is Validation
        :return: None
        """
        import numpy as np
        import os
        import random
        if self.partTarget[-1] != '/' or self.partNeighbour[-1] != '/':
            raise Warning('Folder should end with -> /')
        if val:
            dirListA = os.listdir(self.partValTarget)
        else:
            dirListA = os.listdir(self.partTarget)
        if not val:
            random.shuffle(dirListA)
        if val:
            fileListA = [self.partValTarget + i for i in dirListA]
        else:
            fileListA = [self.partTarget + i for i in dirListA]
        if val:
            neighbourList = [self.partValNeighbour + i for i in dirListA]
            np.save(self.part + 'val.npy', np.array(fileListA))
            print(self.part + 'val.npy', ' Created')
            np.save(self.part + 'valRef.npy', np.array(neighbourList))
            print(self.part + 'valRef.npy', ' Created')
        else:
            neighbourList = [self.partNeighbour + i for i in dirListA]
            np.save(self.part + 'train.npy', np.array(fileListA))
            print(self.part + 'train.npy', ' Created')
            np.save(self.part + 'trainRef.npy', np.array(neighbourList))
            print(self.part + 'trainRef.npy', ' Created')

    def createTestList(self):
        import os
        import numpy as np
        import shutil
        shutil.copytree(self.realOrigin, self.real)
        dirList = os.listdir(self.real)
        fileList = [self.real + i for i in dirList]
        np.save(self.part + 'test.npy', np.array(fileList))
        print(self.part + 'test.npy', ' Created')


if __name__ == '__main__':
    import sys

    part = PartImage2Box()
    try:
        command = sys.argv[1]
    except IndexError:
        print('Command Needed')
        sys.exit()
    # cut pic, file list, cuda
    if command == 'cp':
        if sys.argv[2]:
            b = int(sys.argv[2])
        else:
            b = 500
            s = 64
        if sys.argv[3]:
            s = int(sys.argv[3])
        else:
            s = 64
        part.clean()
        part.cut(b, (s, s))
        print('Preparing Validation')
        part.cut(20, (s, s), isVal=True)

        print('Cut is done \n')
        part.createFileList()
        part.createFileList(val=True)
        part.createTestList()

    elif command == 'fl':
        part.createFileList()
    elif command == 'cuda':
        import torch
        print(torch.cuda.get_device_name(torch.cuda.current_device()))
        print('\n\nIs CUDA Available -> ', torch.cuda.is_available(), '\n\n')

    elif command == 'val':
        part.createFileList(val=True)

    elif command == 'test':
        part.createTestList()

    else:
        print('bad command')
        print('examples:')
        print('tools.py cp cutTimes picSize')
        print('tools.py fl')
        print('tools.py cuda')
        print('tools.py val')
