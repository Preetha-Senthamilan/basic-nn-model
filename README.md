# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
```
DEVELOPED BY : PREETHA S
REGISTER NUMBER : 212222230110
```

## Importing Required packages
```
from google.colab import auth
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import gspread
import pandas as pd
from google.auth import default
import pandas as pd

```
## Authenticate the Google sheet
```
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('Data').sheet1

```
## Split the testing and training data
```
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5,random_state=40)
Scaler = MinMaxScaler()
Scaler.fit(x_train)
X_train1 = Scaler.transform(x_train)

```

## Build the Deep learning Model
```
ai_brain=Sequential([
    Dense(9,activation = 'relu',input_shape=[1]),
    Dense(16,activation = 'relu'),
    Dense(1)
])
ai_brain.compile(optimizer='adam',loss='mse')
ai_brain.fit(X_train1,y_train.astype(np.float32),epochs=2000)

loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()
```
## Evaluate the Model
```
test=Scaler.transform(x_test)
ai_brain.evaluate(test,y_test.astype(np.float32))
n1=[[40]]
n1_1=Scaler.transform(n1)
ai_brain.predict(n1_1)
```

## Dataset Information

![Screenshot 2024-08-19 124400](https://github.com/user-attachments/assets/163d2f78-f96d-4867-ad04-75fb0fac5438)


## OUTPUT

## Training Loss Vs Iteration Plot

![Screenshot 2024-08-19 124844](https://github.com/user-attachments/assets/eb2e4379-fa76-40e8-9535-9af9926f1449)

## Test Data Root Mean Squared Error

![Screenshot 2024-08-19 124943](https://github.com/user-attachments/assets/3044b8f1-5af9-4615-8d94-7c9c69830bc1)


## New Sample Data Prediction

![Screenshot 2024-08-19 125023](https://github.com/user-attachments/assets/ccf98b96-8910-4792-9dbc-6c539a3019bf)


## RESULT

Thus a Neural network for Regression model is Implemented
