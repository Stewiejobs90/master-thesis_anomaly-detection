'''
Created on 31 gen 2018

@author: Matteo Olivato
@author: Omar Cotugno
'''
import csv
from .dataset import Dataset
from .converter import Converter2Image
import numpy as np


class DatasetGenerator(object):
    
    def getVarsNamesDatatypesEntries(self, csv_filename, ignored_fields=[]):
        
        # reading dataset csv file and gets vars name and datatypes
        with open(csv_filename, 'r') as file_reader:
            csv_dset = list(csv.reader(file_reader))
            row_vars_name = csv_dset[0]
            row_datatypes = csv_dset[1]
            dset_entries = csv_dset[2:]

            for field in ignored_fields:
                if field in row_vars_name:
                    i = row_vars_name.index(field)
                    row_vars_name.remove(field)
                    row_datatypes.pop(i)

                    for x in dset_entries:
                        del x[i]
            
        return row_vars_name, row_datatypes, dset_entries
            
    def getMaxNPixDataset(self, csv_filename):
        # creating Converter to Image used for grayscale imgs
        conv = Converter2Image()
        
        row_vars_name, row_datatypes, dset_entries = self.getVarsNamesDatatypesEntries(csv_filename)

        maxn_pix = 0
        for e in dset_entries:
            # retrive pixels conversion of the longest data in the text dataset
            einpix = conv.convert2GrayPixels(e, row_datatypes, row_vars_name)[0]
            npix_side = round(len(einpix)**(1.0/2.0)) + 1
            if npix_side > maxn_pix:
                    maxn_pix = npix_side
        
        return maxn_pix
        
    def genGreyDataset(self, csv_filename, n_pix = None, ignored_fields=[]):
        '''
        Generates a dataset of grayscale images get from csv entries conversion
        labeling and returns them.
        :param csv_filename: csv file containing the dataset
        '''
        # gray scale dataset to return
        gscale_dset_x = []
        
        # creating Converter to Image used for grayscale imgs
        conv = Converter2Image()
                        
        # set the correct pixels number of the side of the squaredimg
        if n_pix != None:
            conv.n_pix = n_pix
            #print(conv.n_pix)
        
        row_vars_name, row_datatypes, dset_entries = self.getVarsNamesDatatypesEntries(csv_filename, ignored_fields)
        
        # converting all dset entries into grayscale imgs
        for e in dset_entries:
            gscale_dset_x.append(conv.convert2GrayImage(e, row_datatypes, row_vars_name))
        
        # return the previously generated dataset converted into numpy array
        return Dataset(np.array(gscale_dset_x))