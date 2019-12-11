import pandas as pd
import os
import numpy as np

dirname = os.path.dirname(__file__)

"""
    File to parse the important data from a raw file.
    Input: None
    Output: None but it writes 2 csv files of positive and negative data
            
    NOTE: THE HEADERS ARE ALSO THERE
"""
def PreProcessData():
    # Read the csv file
    content = pd.read_csv(os.path.join(dirname,"..","..","raw_data","LADECv1-2019.csv"))

    # We only want certain parameters
    column_fields = [
        'c1',
        'c2',
        'stim',
        'nparses',
        'bgJonesMewhort',
        'bgSUBTLEX',
        'bgFacebook',
        'correctParse'
                      ]
    # Apply the filtering
    filtered_content = content.filter(column_fields)

    # We only want correctly parsed entries
    isCorrectParse = filtered_content['correctParse'] == 'yes'

    # apply the filtering
    filteredContentCorrectParse = filtered_content[isCorrectParse]

    # normalize the bigram frequency columns
    filteredContentCorrectParse['bgSUBTLEX'] = (filteredContentCorrectParse['bgSUBTLEX'] - filteredContentCorrectParse['bgSUBTLEX'].mean()) / filteredContentCorrectParse['bgSUBTLEX'].std(ddof=0)
    filteredContentCorrectParse['bgFacebook'] = (filteredContentCorrectParse['bgFacebook'] - filteredContentCorrectParse['bgFacebook'].mean()) / filteredContentCorrectParse['bgFacebook'].std(ddof=0)

    # our positive data are the ones with >= 50 compound rating value
    isPositiveData = filteredContentCorrectParse['ratingcmp'] >= 50

    # our negative data are the ones with < 50 compound rating value
    isNegativeData = filteredContentCorrectParse['ratingcmp'] < 50

    # applying positive and negative filtering
    filteredContentCorrectParse_positive = filteredContentCorrectParse[isPositiveData]
    filteredContentCorrectParse_negative = filteredContentCorrectParse[isNegativeData]

    # Rename the column correctParse to label and change its value from strings to int
    filteredContentCorrectParse_positive = filteredContentCorrectParse_positive.rename(columns={"correctParse": "label"})
    filteredContentCorrectParse_negative = filteredContentCorrectParse_negative.rename(columns={"correctParse": "label"})
    filteredContentCorrectParse_positive['label'] = 1
    filteredContentCorrectParse_negative['label'] = 0

    # # write the outputs to csv files. Index=False to remove adding an index column
    filteredContentCorrectParse_negative.to_csv(dirname,"..","..","pre_processed_data","negative.csv", index=False)
    filteredContentCorrectParse_positive.to_csv(dirname,"..","..","pre_processed_data","positive.csv", index=False)


# Method that reads both positive and negative csv files and returns a tuple of numpy arrays of both datasets
def CSV2Numpy():

    # Use pandas to read the files and output them as n-dimensional numpy arrays ignoring the constituents as well as
    # the compound words themselves
    positiveArray = pd.read_csv(os.path.join(dirname,'..','..',"pre_processed_data","positive_final.csv"), skiprows=1).values[:, 3:]
    negativeArray = pd.read_csv(os.path.join(dirname,'..','..',"pre_processed_data","negative_final.csv"), skiprows=1).values[:, 3:]

    return positiveArray, negativeArray

"""
    Method that returns a shuffled dataSet
"""
def preProcess_pipeline():
    positive, negative = CSV2Numpy()
    dataSet = np.concatenate((positive, negative), 0)
    np.random.shuffle(dataSet)
    return dataSet

if __name__ == "__main__":
    pass
    # PreProcessData()
    # x = preProcess_pipeline()