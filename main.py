import kaggle
import tensorflow as tf
import numpy as np
import pandas
import csv
import json

from sklearn.preprocessing import normalize
from tensorflow.python.keras import layers

def main(dataDirectory):
  # Preprocessed training data
  data = prepData(dataDirectory, True)
  print(data)

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
  if (trainBoolean):
    dataDirPrefix = 'train_'
  else:
    dataDirPrefix = 'test_'
  # We will ultimately have 677 attributes for each sample,
  # each with a value between 0 and 1.
  processedData = np.zeros((data.shape[0], 677))
  # Iterate over the columns of the data, keeping track of where we are
  for index, col in enumerate(data.transpose()):
    
    '''
    Looking at type attribute (dog or cat).
    Transform it into two boolean attributes.
    '''
    if (index == 0):
      # Creating new attribute array
      newAttribs = np.zeros((processedData.shape[0], 2))
      # Fill new attribute array
      for ind, sample in enumerate(col):
        # Dog is first attribute, cat is second attribute
        newAttribs[ind][sample-1] = 1
      # Save the new attributes to processedData
      processedData[:, :2] = newAttribs
    
    '''
    Looking at the name attribute.
    Max normalized, as represented by a 0-1 range float.
    '''
    if (index == 1):
      # Calculate length of each name
      lengthF = lambda x: len(str(x))
      newAttribs = np.array(list(map(lengthF, col)))
      newAttribs = normalize(newAttribs.reshape(-1, 1), norm='max', axis=0)
      newAttribs = newAttribs.reshape((1, newAttribs.shape[0]))
      # Save new attributes to processedData
      processedData[:, 2] = newAttribs

    '''
    Looking at the age attribute.
    Normalized by 216 months (dogs don't really live over 18 years).
    '''
    if (index == 2):
      normF = lambda x: x/216
      newAttribs = np.array(list(map(normF, col)))
      processedData[:, 3] = newAttribs

    '''
    Looking at Breed1 attribute.
    There are 306 different breeds, 307 including
    mixed breeds. We need 307 new boolean attributes to represent this.
    '''
    if (index == 3):
      # Creating new attribute array
      newAttribs = np.zeros((processedData.shape[0], 307))
      # Fill new attribute array
      for ind, sample in enumerate(col):
        # Same logic as type attribute (dog, cat)
        newAttribs[ind][sample-1] = 1
      # Save the new attributes to processedData
      processedData[:, 4:311] = newAttribs
    
    '''
    Looking at Breed2 attribute.
    Same logic as Breed2 attribute, this time with a possible 0 value
    for non-entry 
    '''
    if (index == 4):
      newAttribs = np.zeros((processedData.shape[0], 308))
      for ind, sample in enumerate(col):
        # Notice sample no longer -1 because it can take on 0 value
        newAttribs[ind][sample] = 1
      processedData[:, 311:619] = newAttribs

    '''
    Looking at Gender attribute.
    Same logic as type attribute.
    '''
    if (index == 5):
      newAttribs = np.zeros((processedData.shape[0], 3))
      for ind, sample in enumerate(col):
        newAttribs[ind][sample-1] = 1
      processedData[:, 619:622] = newAttribs

    '''
    Looking at Color1 attribute.
    Same logic as type attribute.
    '''
    if (index == 6):
      newAttribs = np.zeros((processedData.shape[0], 7))
      for ind, sample in enumerate(col):
        newAttribs[ind][sample-1] = 1
      processedData[:, 622:629] = newAttribs
    
    '''
    Looking at Color2 attribute.
    Same logic as Color1 attriubte, this time with a possible 0 value
    for non-entry.
    '''
    if (index == 7):
      newAttribs = np.zeros(processedData.shape[0], 8)
      for ind, sample in enumerate(col):
        newAttribs[ind][sample] = 1
      processedData[:, 629:637] = newAttribs

    '''
    Looking at Color3 attribute.
    Same logic as Color 2
    '''
    if (index == 8):
      newAttribs = np.zeros(processedData.shape[0], 8)
      for ind, sample in enumerate(col):
        newAttribs[ind][sample] = 1
      processedData[:, 637:645] = newAttribs

    '''
    Looking at MaturitySize attribute.
    This can take on 5 values. (0 indexed)
    '''
    if (index == 9):
      newAttribs = np.zeros(processedData.shape[0], 5)
      for ind, sample in enumerate(col):
        newAttribs[ind][sample] = 1
      processedData[:, 645:650] = newAttribs

    '''
    Looking at FurLength attribute.
    This can take on 4 values. (0 indexed)
    '''
    if (index == 10):
      newAttribs = np.zeros(processedData.shape[0], 4)
      for ind, sample in enumerate(col):
        newAttribs[ind][sample] = 1
      processedData[:, 650:654] = newAttribs
    
    '''
    Looking at vaccinated attribute.
    This can take on 3 values. (1 indexed)
    '''
    if (index == 11):
      newAttribs = np.zeros(processedData.shape[0], 3)
      for ind, sample in enumerate(col):
        newAttribs[ind][sample-1] = 1
      processedData[:, 654:657] = newAttribs

    '''
    Looking at Dewormed attribute.
    This can take on 3 values. (1 indexed)
    '''
    if (index == 12):
      newAttribs = np.zeros(processedData.shape[0], 3)
      for ind, sample in enumerate(col):
        newAttribs[ind][sample-1] = 1
      processedData[:, 654:657] = newAttribs

    '''
    Looking at Sterilized attribute.
    This can take on 3 values. (1 indexed)
    '''
    if (index == 13):
      newAttribs = np.zeros(processedData.shape[0], 3)
      for ind, sample in enumerate(col):
        newAttribs[ind][sample-1] = 1
      processedData[:, 657:660] = newAttribs

    '''
    Looking at Health attribute.
    This can take on 4 values. (0 indexed)
    '''
    if (index == 14):
      newAttribs = np.zeros(processedData.shape[0], 4)
      for ind, sample in enumerate(col):
        newAttribs[ind][sample] = 1
      processedData[:, 660:654] = newAttribs
    
    '''
    Looking at Quantity attribute.
    Max normalized as represented by a 0-1 float range
    '''
    if (index == 15):
      newAttribs = normalize(col.reshape(-1, 1), norm='max', axis=0)
      newAttribs = newAttribs.reshape((1, newAttribs.shape[0]))
      # Save new attributes to processedData
      processedData[:, 654] = newAttribs

    '''
    Looking at Fee attribute.
    Normalize by 2000, I don't think a fee will ever exceed this value.
    '''
    if (index == 16):
      normF = lambda x: x/2000
      newAttribs = np.array(list(map(normF, col)))
      processedData[:, 655] = newAttribs

    '''
    Looking at State attribute.
    Cannot use state id as index because it is not 0 or 1 based.
    '''
    if (index == 17):
      # Create map from state_labels to 0 based indices
      mapping = {}
      reader = csv.reader(open('./breeder_labels.csv', 'r'))
      # First index is label
      for ind, row in reader:
        if (ind == 0):
          continue
        k, _ = row
        mapping[k] = ind - 1 
      # There should be mapping length new attributes
      newAttribs = np.zeros(processedData.shape[0], 15)
      for ind, sample in enumerate(col):
        # Activate the attribute corresponding to the state for this sample
        newAttribs[ind][mapping[sample]] = 1
      processedData[:, 656:671]

    '''
    Looking at VideoAmt attribute.
    Normalized by 10, likely no more than 10 videos per pet
    '''
    if (index == 18):
      normF = lambda x: x/10
      newAttribs = np.array(list(map(normF, col)))
      processedData[:, 671] = newAttribs
    
    '''
    Looking at Description attribute.
    Max normalized as represented by a 0-1 range float
    '''
    if (index == 19):
      # Calculate length of each description
      lengthF = lambda x: len(str(x))
      newAttribs = np.array(list(map(lengthF, col)))
      newAttribs = normalize(newAttribs.reshape(-1, 1), norm='max', axis=0)
      newAttribs = newAttribs.reshape((1, newAttribs.shape[0]))
      processedData[:, 672] = newAttribs

    '''
    Looking at PetID attribute.
    Use it to generate color and sentiment attributes
    '''
    if (index == 20):
      # Generate color attributes (3 different colors between 0-1)
      newAttribs = np.zeros(processedData.shape[0], 3)
      for ind, sample in enumerate(col):
        # Find associated metadata color with highest score
        r, g, b = 0
        with open('./data' + dataDirPrefix + 'metadata' + sample + '.json') as json_file:
          metadata = json.load(json_file)
          currentHighestScore = 0
          for item in metadata['imagePropertiesAnnotation']['dominantColors']['colors']:
            if (item['score'] > currentHighestScore):
              r = item['color']['red']
              g = item['color']['green']
              b = item['color']['blue']
              currentHighestScore = item['score']
        r /= 255
        g /= 255
        b /= 255
        newAttribs[ind] = [r, g, b]
      processedData[:, 673:676] = newAttribs

      # Generate sentiment attribute (1 attribute)
      newAttribs = np.zeros(processedData.shape[0], 1)
      for ind, sample in enumerate(col):
        # Find associated average sentiment
        sentimentVal = 0
        sentimentDivisor
        with open('./data' + dataDirPrefix + 'sentiment' + sample + '.json') as json_file:
          sentiment = json.load(json_file)
          for item in sentiment['sentences']['sentences']:
            sentimentVal += item['magnitude'] * item['score']
          sentimentDivisor = len(sentiment['sentences'])
        sentimentVal /= sentimentDivisor
        newAttribs[ind] = sentimentVal
      processedData[:, 676] = newAttribs

    '''
    Looking at PhotoAmt attribute.
    I think most important aspect of this attribute is photo absence, so
    Max normalization on any distribution with non-zero elements should do.
    '''
    if (index == 21):
      newAttribs = normalize(col.reshape(-1, 1), norm='max', axis=0)
      newAttribs = newAttribs.reshape((1, newAttribs.shape[0]))
      # Save new attributes to processedData
      processedData[:, 677] = newAttribs

  return processedData


def readCoreData(dataDirectory, trainBoolean):
  if (trainBoolean):
    # Returns data in numpy array form without column labels
    data = pandas.read_csv(dataDirectory + '/train.csv').to_numpy()
  else:
    data = pandas.read_csv(dataDirectory + '/test/test.csv').to_numpy()
  data = removeUnusedColumns(data)
  return data

'''
removeUnusedColumns() should remove colums from the data which
are not going to be used in the predictor.
'''
def removeUnusedColumns(data):
  '''
  Removing rescuerID because it sholdn't
  matter. This indice is 18 (0 based indexing).
  '''
  return np.delete(data, [18], axis=1)

'''
Function creates the model for predicting adoption speed
'''
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