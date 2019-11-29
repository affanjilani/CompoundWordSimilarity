from sklearn.linear_model import LogisticRegression

# Preprocess the data

# Create the model(s)

logReg = LogisticRegression(solver='lbfgs')



# Split into training and evaluation

# Train the models
# Assume x_train, y_train, x_test, y_test
logReg.fit(x_train,y_train)

# Evaluate the models using the validation set
print(logReg.score(x_test,y_test))