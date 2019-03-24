import kaggle
import tensorflow as tf
import numpy as np

from tensorflow.python.keras import layers

def main(dataDirectory):
  print("in main")
  # Preprocessed training data
  data = prepData(dataDirectory, True)

'''
prepData() should return numpy array with input shape
(n,m), where n is the number of samples and m is the
number of derived attributes. Each attribute in m
should be normalized to the range (0-1) so that no
attribute is considered significantly more important
than another.
'''
def prepData(dataDirectory, trainBoolean):
  data = readCoreData(dataDirectory, trainBoolean)

def readCoreData(dataDirectory, trainBoolean):
  if (trainBoolean):
    data = np.genfromtxt(dataDirectory + 'train.csv', delimiter=',')
  else:
    data = np.genfromtxt(dataDirectory + '/test/test.csv', delimiter=',')
  # Removing column labels
  data = data[1:, :]
  return data

'''
removeUnusedColumns() should remove colums from the data which
are not going to be used in the predictor.
'''
def removeUnusedColumns(data):
  '''
  Removing name, rescuerID because they intuitively don't
  matter. TODO name size may be able to extract useful information
  These indices are 1, 18
  '''
  np.delete()



def createModel():
  model = tf.keras.Sequential()


'''
Will only work if new entrants are not prohibited.
Must have a kaggle account and follow instructions for API token @
https://github.com/Kaggle/kaggle-api under the API credentials section.

I found another copy of the data @ https://github.com/qemtek/PetFinder
which can be downloaded manually.
'''
def downloadData(path):
  kaggle.api.authenticate()
  kaggle.api.competition_download_files('petfinder-adoption-prediction', path)

'''
Parameters:
dataDirectory
'''
main('./data')