import pandas as pd
import os

"""
    Method to parse the important data and outputs a positive and negative datasets.
    Input: None
    Output: positive.csv --> positive labelled data
            negative.csv --> negative labelled data
            
    NOTE: THE HEADERS ARE ALSO THERE
"""
def ParseData():
    # Read the csv file
    content = pd.read_csv(os.path.join("../../raw_data/LADECv1-2019.csv"))

    # We only want certain parameters
    column_fields = [
        'c1',
        'c2',
        'stim',
        'nparses',
        'ratingcmp',
        'ratingC1',
        'ratingC2',
        'bgJonesMewhort',
        'bgSUBTLEX',
        'bgFacebook',
        'LSAc1c2',
        'LSAc1stim',
        'LSAc2stim',
        'c1c2_snautCos',
        'c1stim_snautCos',
        'c2stim_snautCos',
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


    # # write the outputs to csv files. Index=False to remove adding an index column
    pd.DataFrame(filteredContentCorrectParse_negative).to_csv("../../pre_processed_data/negative.csv", index=False)
    pd.DataFrame(filteredContentCorrectParse_positive).to_csv("../../pre_processed_data/positive.csv", index=False)

if __name__ == "__main__":
    ParseData()