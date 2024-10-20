import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_csv("C:/Users/IsoCom/Desktop/kodi/PunimiFinalML/dataset.csv", encoding='ISO-8859-1')
train_data = train_data.fillna('')

x = np.array(train_data["Text"])
y = np.array(train_data["language"])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

cv = CountVectorizer()
X_train = cv.fit_transform(x_train)
X_test = cv.transform(x_test)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

user = input("Jepni nje tekst: ")
data = cv.transform([user]).toarray()
output = model.predict(data)
print("Prediction:", output)

target_names = list(reversed(train_data['language'].unique()))
print(classification_report(y_test, y_pred, target_names=target_names))


confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
#print(confusion_matrix)
#confusion_matrix NUK E KEMI PRINTUAR PER ARSYE ESHTE SHUME E GJATE 

sns.heatmap(confusion_matrix, annot=True, fmt='d')

labels = train_data['language'].unique()
num_labels = len(labels)
plt.xticks(range(num_labels), labels[::-1], rotation=90)
plt.yticks(range(num_labels), labels[::-1], rotation=0)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
