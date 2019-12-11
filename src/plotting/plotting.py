import matplotlib.pyplot as plt

"""
    Method that plots the data
    Input: 
        accuracies --> accuracy scores (list of floats). this is our y-axis
        hyperparameters --> The respective hyperparameters used (list of lists of strings). this is our x-axis
        e.g. [["SF","l"],["i","l"]]
"""
def plottingData(accuracies, hyperparameters):
    # creating a list of strings representing our x-axis labels
    xLabels = []
    for lst in hyperparameters:
        string = ""
        for str in lst:
            string += str + ", "

    pass







if __name__ == "__main__":
    # x-coordinates of left sides of bars
    left = ['SF', "S", "F", "SP", "SM"]

    # heights of bars
    height = [10, 24, 36, 40, 5]

    # labels for bars
    # tick_label = ['one', 'two', 'three', 'four', 'five']

    # plotting a bar chart
    plt.bar(left, height,
            width=0.8, color=['red', 'green'])

    # naming the x-axis
    plt.xlabel('x - axis')
    # naming the y-axis
    plt.ylabel('y - axis')
    # plot title
    plt.title('My bar chart!')

    # function to show the plot
    plt.show()