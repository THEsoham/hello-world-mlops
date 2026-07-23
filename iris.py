from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['flower_name'] = df['target'].map({0:'setosa', 1:'versicolor', 2:'virginica'})
print(df.head(10))