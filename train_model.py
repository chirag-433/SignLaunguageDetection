import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Map directory numbers to actual letters for display
labels_dict = {'0': 'A', '1': 'B', '2': 'L', '3': 'C', '4': 'D', '5': 'E', '6': 'F'}
unique_labels = np.unique(labels)
trained_letters = [labels_dict.get(str(lbl), str(lbl)) for lbl in unique_labels]
print(f"Training model on the following letters: {', '.join(trained_letters)}")

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
