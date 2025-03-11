# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary libraries.
2. Analyse and if needed preprocess the dataset.
3. Assign the features dor the model.
4. Train, test and split the data.
5. Implement the algorithm using LinearRegression().
6. Test the model.
7. Measure the performance metrics of the ML model.
8. Plot the graph.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: JAYANI N
RegisterNumber:  24900024
*/
```

    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_absolute_error,mean_squared_error
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    
    dataset=pd.read_csv('/content/student_scores.csv')
    print(dataset.head())
    print(dataset.tail())
    dataset.info()
    
    x=dataset.iloc[:,:-1].values #starts from first untill the last before column
    print(x)
    y=dataset.iloc[:,1].values #only the last column is extracted
    print(y)
    print(x.shape)
    print(y.shape)
    
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    
    reg=LinearRegression()
    reg.fit(x_train,y_train)
    y_pred=reg.predict(x_test)
    print(y_pred)
    print(y_test)
    
    mse=mean_squared_error(y_test,y_pred)
    print('MSE = ',mse)
    mae=mean_absolute_error(y_test,y_pred)
    print('MAE = ',mae)
    rmse=np.sqrt(mse)
    print('RMSE = ',rmse)
    
    plt.scatter(x_test,y_test,color='blue')
    plt.plot(x_train,reg.predict(x_train),color='green')
    plt.show

## Output:

![image](https://github.com/user-attachments/assets/25fe87ba-41de-41f7-9238-052fae0a3f36)
![image](https://github.com/user-attachments/assets/feaad1a6-81a5-411f-a95d-2939ce58a84c)
![image](https://github.com/user-attachments/assets/2cd36721-0d8a-4b0f-90bd-f961cebee1f4)
![image](https://github.com/user-attachments/assets/f20eeccb-a384-467a-bb07-0df14d434a88)
![image](https://github.com/user-attachments/assets/f780158f-2890-4931-b6d0-023ca592e645)
![image](https://github.com/user-attachments/assets/94556481-729d-47ca-97a6-fec3cf405b40)
![image](https://github.com/user-attachments/assets/8a8de946-03f0-44d6-9cd0-4a8973577efc)
![image](https://github.com/user-attachments/assets/b6be2b27-d21a-4a6e-93bf-d5dc1b7af569)






## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
