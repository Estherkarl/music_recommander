import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load data
data = pd.read_csv('data.csv')

# Encode users and songs
user_encoder = LabelEncoder()
song_encoder = LabelEncoder()

data['user'] = user_encoder.fit_transform(data['user_id'])
data['song'] = song_encoder.fit_transform(data['song_id'])

# Split data
X = data[['user', 'song']]
y = data['listen_count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
