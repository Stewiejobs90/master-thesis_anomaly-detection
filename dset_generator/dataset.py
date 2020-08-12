'''
Created on 03 feb 2018

@author: Matteo Olivato
@author: Omar Cotugno
'''
import random
import numpy as np
import matplotlib.pyplot as plt


class Dataset(object):
    '''
    Class representing dataset object of imgs generated from csv log file
    '''

    def __init__(self, dset_x = None, is_gray = True):
        '''
        Constructor
        '''
        
        self.dset_x = dset_x
        self.is_gray = is_gray
        
    def __show_img(self, img):
        '''
        Show a generated grayscale img with matplotlib
        :param img: img to show
        '''
        
        if self.is_gray:
            plt.imshow(img, cmap = 'gray', interpolation = 'none')
        else:
            plt.imshow(img, cmap = 'rgba', interpolation = 'none')

        plt.show()
    
    def __save_img(self, img, path, imgname):
        '''
        Save a generated grayscale img with matplotlib
        :param img: img to save
        '''
        imgpath = path + imgname + '.png'
        
        if self.is_gray:
            plt.imshow(img, cmap='gray', interpolation='none')
        else:
            plt.imshow(img, cmap='rgba', interpolation='none')
        
        plt.savefig(imgpath)
        plt.close()
    
    def join(self, dsets):
        '''
        Join a list of datsets into one datset.
        Generally useful when you need to join datasets with different labels
        :param dsets:
        '''
        # joining all dsets in the list in one numpy array
        for i in dsets:
            if self.is_gray == i.is_gray:
                self.dset_x = np.append(self.dset_x, i.dset_x, axis=0)
            else:
                raise ValueError('Different color types of joinig datasets.')
            
    def save_npz(self, npz_filename):
        '''
        Save dataset into npz (numpy compressed) format
        :param dset: dataset to save compressed
        '''
        np.savez_compressed(npz_filename, x=self.dset_x, is_gray=self.is_gray)
        
    def load_npz(self, npz_filename, is_gray = True):
        '''
        Load dataset saved into npz (numpy compressed) format
        :param npz_filename: dataset filename to load
        :param is_gray: boolean value indicates if the dset's immages are in grayscale (default True)
        '''
        f = np.load(npz_filename)
        self.is_gray = f['is_gray']
        self.dset_x = f['x']
    
    def shuffle(self):
        '''
        Shuffles all datasets entries mantaining correct data -> label association
        '''
        # create a sigle dataset
        dset_x = self.dset_x.tolist()
        
        # shuffle all elements
        random.shuffle(dset_x)
        
        # redefine dset_x
        self.dset_x = np.array(dset_x).astype(np.uint8)
    
    def show(self, nmax=-1, randomly=False):
        '''
        Shows imgs saved into a dataset using matplotlib
        :param nmax: maximum number of imgs to show
        :param randomly: chooses if the imgs are choosen randomly from the dataset
        '''
        # if nmax is default set namx into len(dset)
        if nmax == -1:
            nmax = len(self.dset)
        
        # if you want to show imgs choosen randomly from the dataset
        if randomly:
            rimgs = random.sample(range(len(self.dset_x)), nmax)
        else:
            rimgs = range(nmax)
            
        # show all imgs in the range rimgs
        for i in rimgs:
            self.__show_img(self.dset_x[i])
    
    def saveImgs(self, path, imgname, nmax=-1, randomly=False):
        '''
        Save imgs from a dataset using matplotlib
        :param nmax: maximum number of imgs to save
        :param randomly: chooses if the imgs are choosen randomly from the dataset
        '''
        # if nmax is default set nmax into len(dset)
        if nmax == -1:
            nmax = len(self.dset)
        
        # if you want to show imgs choosen randomly from the dataset
        if randomly:
            rimgs = random.sample(range(len(self.dset_x)), nmax)
        else:
            rimgs = range(nmax)
            
        # save all imgs in the range rimgs
        for i in rimgs:
            self.__save_img(self.dset_x[i], path, imgname + '_' + str(i))