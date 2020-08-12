'''
Created on 02 feb 2019

@author: Matteo Olivato
@author: Omar Cotugno
'''

from .dataset_generator import DatasetGenerator
import os


class DsetManager(object):
    """docstring for DsetManager"""

    def __init__(self, outpath, shuffle = True):
        """
        Constructor
        """
        self.outpath = outpath
        # create a DatasetGenerator
        self.dg = DatasetGenerator()
        self.shuffle = shuffle

    def saveDataset(self, namedset, dset):
        # creates outpath if it not exists
        if not os.path.exists(self.outpath):
            os.mkdir(self.outpath)

        path = self.outpath if self.outpath.endswith('/') else self.outpath + '/'

        # save dataset as npz file
        dset.save_npz(path + namedset)

    def generateDataset(self, path, dset_name, n_pix = None, ignored_fields=[]):

        max_side_pixel_numbers = self.dg.getMaxNPixDataset(path)

        if n_pix:
            if n_pix < max_side_pixel_numbers:
                print('WARN: n_pix < max_side_pixel_numbers -> Possible loss of informations...')
            max_side_pixel_numbers = n_pix

        #print('Using n_pix = %d' % max_side_pixel_numbers)

        dset = self.dg.genGreyDataset(path, max_side_pixel_numbers, ignored_fields)

        if self.shuffle:
            dset.shuffle()

        self.saveDataset(dset_name, dset)
