import numpy as np
import pandas as pd

# Takes in the original data array with label and adds the augmented features
def augment_data(data, augmentation):
    init_features = np.array(data[:,:-1])

    # Turn labels into column
    labels = np.array(data[:,-1]).reshape(-1,1) 

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

"""
    Method that does log transformation on the data
    Input: NxM matrix of data as numpy array
    Output: Nx(M + (M-2))
"""
def log_transformation(data):
    # conver the numpy into a data frame
    df = pd.DataFrame(data)
    for i in df.columns:
        # ignore the nparses and label columns as their transformation is irrelavent
        if i == 0 or i == 7:
            continue
        # Create our new data frame column
        columnOfInterest = df[i]
        # a numpy column vector
        newColumn = columnOfInterest.to_numpy(dtype=np.float32).reshape(-1, 1)
        # apply log transformation handling negative values
        newColumn = np.log((newColumn - np.min(newColumn)) + 1)
        # apply log transformation and append it to the data frame
        df.insert(6+i, str(i)+"_log", newColumn)

    return df.values


if __name__ == '__main__':
    pass
    # test = np.arange(0,12).reshape(3,4)
    # print(test)
    # print(first_order_interactions(test))
    # print(augment_data(test,first_order_interactions(test)))
