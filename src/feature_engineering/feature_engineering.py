import numpy as np
import pandas as pd
from src.pre_processing.preProcess import CSV2Numpy

# Takes in the original data array with label and adds the augmented features
def augment_data(data, augmentation):
    init_features = np.array(data[:,:-1])

    # Turn labels into column
    labels = np.array(data[:,-1]).reshape(-1, 1)

    # Insert the augmentation
    features = np.hstack((init_features,augmentation))

    # Add the labels back
    return np.hstack((features,labels))


# Takes as input a nxm array.
# Returns a nxm array of all first order interactions of the features
def first_order_interactions(data):

    #copy the data
    data = np.array(data)

    features = data[:,:-1]

    initialSize = np.shape(features)[1]

    newFeatures = None

    for index in range(initialSize):
        index2 = index
        print(np.shape(features))
        while index2<initialSize:
            col1 = features[:,index]
            col2 = features[:,index2]

            #multiply them together to get interaction column
            col12 = np.multiply(col1,col2)
            #reshape them into a column array
            col12 = np.reshape(col12,(-1,1))

            if newFeatures is None:
                newFeatures = col12
            else:
                newFeatures = np.hstack((newFeatures,col12))

            index2+=1
        
    return newFeatures

# Method that performs log transformation on only similarities.
# category: either S=similarities,F=frequencies,SF=similarities and frequencies
def log_transformation(data, categoryToTransform):
    # get the shape of matrix
    row, col = np.shape(data)
    # retrieve the last col
    col = col - 1
    columnsToIgnore = []
    # pick certain columns based on category
    if categoryToTransform.lower() == "s":
        columnsToIgnore = [0,1,2,3,col]
    elif categoryToTransform.lower() == "f":
        columnsToIgnore = [0,4,5,6,col]
    elif categoryToTransform.lower() == "sf":
        columnsToIgnore = [0,col]
    # Create a dataframe
    new_df = pd.DataFrame()
    df = pd.DataFrame(data)
    for i in df.columns:
        # ignore some columns as they are irrelevant
        if i in columnsToIgnore:
            continue
        # Create our new data frame column
        columnOfInterest = df[i]
        # a numpy column vector
        newColumn = columnOfInterest.to_numpy(dtype=np.float32)
        # apply log transformation handling negative values
        newColumn = np.log((newColumn - np.min(newColumn)) + 1)
        # newColumn = np.log(newColumn  + 1)
        # apply log transformation and append it to the data frame
        new_df[i] = newColumn
    # return the array containing the transformation logs
    return new_df.values

"""
    Method that combines all the possible feature engineering that we want and outputs the desired final dataset
    Input: 
        Data: our original dataset
        categoryToTransform: will be used in log transformation. see log_transformation method
        ListOfFeatureEngineering: list of strings that our data set will undergo. e.g: ['I','L'] --> 
        interactions and log transformation. NOTE: ORDER IS IMPORTANT
        isAugmentAfterEachFeatureEngineer: boolean whether or not we augment after every feature engineering
"""
def feature_engineering_pipeline(data, categoryToTransform, listOfFeatureEngineering, isAugmentAfterEachFeatureEngineer):
    # our final list of things to augment
    listOfDataToAugment = []

    # Iterate through the list and apply the transformations
    for string in listOfFeatureEngineering:
        if string.lower() == 'i':
            # interactions, check our flag
            if isAugmentAfterEachFeatureEngineer:
                data = augment_data(data,first_order_interactions(data))
            else:
                listOfDataToAugment.append(first_order_interactions(data))
        elif string.lower() == 'l':
            # log transformation
            if isAugmentAfterEachFeatureEngineer:
                data = augment_data(data,log_transformation(data,categoryToTransform))
            else:
                listOfDataToAugment.append(log_transformation(data,categoryToTransform))

    # finally augment everything
    for dataToAugment in listOfDataToAugment:
        data = augment_data(data,dataToAugment)

    # return the final data
    return data

if __name__ == '__main__':
    # pos,neg = CSV2Numpy()
    # dataSet = np.concatenate((pos, neg), 0)
    # feature_engineering_pipeline(dataSet, 'SF', ['l','i'], True)
    pass