'''
Created on 30 gen 2018

@author: Matteo Olivato
@author: Omar Cotugno
'''
import bitstring as bs
import numpy as np
from numpy import uint8


class Converter2Image(object):
    '''
    Class that converts  a csv file entry to 8 bits grayscale squared img.
    We have fixed length for types int and float.
    Strings type length isn't fixed.
    We can converts images into two types:
        - img from a stream of pixel
        - img combining subimgs created dividing entry into different semantics blocks
    '''
    
    def __init__(self, n_pix = 32, int_len = 64, float_len = 64, subimgs_filters = ['']):
        '''
        Constructor
                
        :param n_pix: side number of final squared grayscale img
        :param int_len: fixed bits length for int representation
        :param float_len: fixed bits length for float representation
        :param subimgs_filters: filter for vars names in order to produce an subimgs composed img
        '''
        
        self.n_pix = n_pix
        self.int_len = int_len
        self.float_len = float_len
        self.subimgs_filters = subimgs_filters
    
    def __int2pixels(self, x):
        '''
        Converts integer number into fixed length (0 to 255) int list
        :param x: integer number to convert
        '''
        
        # gets string representation of the integer x with fixed 64 bits length
        l = bs.BitString(int = x, length = 64).bin
        # returns a list of integers after spitting bit string in 8bits chunks
        return [ int(l[i:i + 8], 2) for i in range(0, len(l), 8) ]

    def __float2pixels(self, x):
        '''
        Converts float number into fixed length (0 to 255) int list
        :param x: float number to convert
        '''
        
        # gets string representation of the float x with fixed 64 bits length
        l = bs.BitString(float = x, length = 64).bin
        # returns a list of integers after spitting bit string in 8bits chunks
        return [ int(l[i:i + 8], 2) for i in range(0, len(l), 8) ]
    
    def __str2pixels(self, s):
        '''
        Converts string into (0 to 255) int list using ord()
        :param s: string to convert
        '''
        return [ord(c) for c in s]
    
    def convert2GrayPixels(self, row_data, row_datatypes, row_vars_name = ['']):
        '''
        Converts a csv file entry into a list of list of grascale pixels
        :param row_data: csv entry
        :param row_datatypes: csv entry data colums types
        :param row_var_names: names of var related to entry vals
        '''
        # if no var names passed using data types
        if row_vars_name[0] == '':
            row_vars_name = row_datatypes 
        
        pixs = []
        # scan differents filters
        for sif in self.subimgs_filters:
            pix = []
            # scan data to convert
            for i in range(0, len(row_data)):
                # if this filter is into var name
                if sif in row_vars_name[i]:
                    # converts using the correct type
                    if ("double" in row_datatypes[i]):
                        pix += self.__float2pixels(float(row_data[i]))
                    elif ("int" in row_datatypes[i]):
                        pix += self.__int2pixels(int(row_data[i]))
                    else:
                        pix += self.__str2pixels(row_data[i])
            # appends creating a list of list of pixel
            pixs.append(pix)
            
        return pixs

    def convert2GrayImage(self, row_data, row_datatypes, row_vars_name = ''):
        '''
        Converts a csv file entry into a grayscale img
        :param row_data: csv entry
        :param row_datatypes: csv entry data colums types
        :param row_var_names: names of var related to entry vals
        '''
        # convert row data (a csv entry) into gray pixels
        pixels = self.convert2GrayPixels(row_data, row_datatypes, row_vars_name)
        
        # checks if pixels area is more than n_pix^2 value (pixels can't fit into this img)
        if round(len(pixels[0])**(1.0/2.0)) + 1 > self.n_pix :
            raise ValueError('The number of pixels exceed img size. Try to use: '+ str(round(len(pixels[0]) ** (1.0/2.0)) + 1) + ' as n_pix') 
        
        # coverts pixels into a numpy array of 8 bits unsigned int 
        gray_img = np.array(pixels[0], dtype = uint8)
        # adding padding for correct reshaping
        gray_img = np.pad(gray_img, (0, (self.n_pix ** 2 - len(gray_img)) ), 'constant', constant_values = (0))
        
        # returns a matrix n_pix x n_pix
        return np.reshape(gray_img, (self.n_pix, self.n_pix))