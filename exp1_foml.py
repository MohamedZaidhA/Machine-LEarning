
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

url = 'https://raw.githubusercontent.com/MohamedZaidhA/Machine-LEarning/21eb44c703b53ce7014cd008e67fe34d33d80caf/Iris.csv'
df = pd.read_csv(url)
display(df.head())
display(df.shape)

#univariate for sepal width
df_Iris_setosa=df.loc[df['Species']=='Iris-setosa']
df_Virginica=df.loc[df['Species']=='Iris-virginica']
df_Versicolor=df.loc[df['Species']=='Iris-versicolor']
plt.scatter(df_Iris_setosa['SepalWidthCm'],np.zeros_like(df_Iris_setosa['SepalWidthCm']))
plt.scatter(df_Virginica['SepalWidthCm'],np.zeros_like(df_Virginica['SepalWidthCm']))
plt.scatter(df_Versicolor['SepalWidthCm'],np.zeros_like(df_Versicolor['SepalWidthCm']))
plt.xlabel('SepalWidthCm')
plt.show()

#univariate for sepal length
df_Iris_setosa=df.loc[df['Species']=='Iris-setosa']
df_Virginica=df.loc[df['Species']=='Iris-virginica']
df_Versicolor=df.loc[df['Species']=='Iris-versicolor']
plt.scatter(df_Iris_setosa['SepalLengthCm'],np.zeros_like(df_Iris_setosa['SepalLengthCm']))
plt.scatter(df_Virginica['SepalLengthCm'],np.zeros_like(df_Virginica['SepalLengthCm']))
plt.scatter(df_Versicolor['SepalLengthCm'],np.zeros_like(df_Versicolor['SepalLengthCm']))
plt.xlabel('SepalLengthCm')
plt.show()

#univariate for petal width
df_Setosa=df.loc[df['Species']=='Iris-setosa']
df_Virginica=df.loc[df['Species']=='Iris-virginica']
df_Versicolor=df.loc[df['Species']=='Iris-versicolor']
plt.scatter(df_Setosa['PetalWidthCm'],np.zeros_like(df_Setosa['PetalWidthCm']))
plt.scatter(df_Virginica['PetalWidthCm'],np.zeros_like(df_Virginica['PetalWidthCm']))
plt.scatter(df_Versicolor['PetalWidthCm'],np.zeros_like(df_Versicolor['PetalWidthCm']))
plt.xlabel('PetalWidthCm')
plt.show()

#univariate for petal length

df_Setosa=df.loc[df['Species']=='Iris-setosa']
df_Virginica=df.loc[df['Species']=='Iris-virginica']
df_Versicolor=df.loc[df['Species']=='Iris-versicolor']
plt.scatter(df_Setosa['PetalLengthCm'],np.zeros_like(df_Setosa['PetalLengthCm']))
plt.scatter(df_Virginica['PetalLengthCm'],np.zeros_like(df_Virginica['PetalLengthCm']))
plt.scatter(df_Versicolor['PetalLengthCm'],np.zeros_like(df_Versicolor['PetalLengthCm']))
plt.xlabel('PetalLengthCm')
plt.show()

#bivariate sepal.width vs petal.width
sns.FacetGrid(df,hue='Species',height=5).map(plt.scatter,"SepalWidthCm","PetalWidthCm").add_legend();
#bivariate sepal.length vs petal.length
sns.FacetGrid(df,hue='Species',height=5).map(plt.scatter,"SepalLengthCm","PetalLengthCm").add_legend();
plt.show()

#multivariate all the features
sns.pairplot(df,hue="Species",height=2)
