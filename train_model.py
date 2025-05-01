import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

data={
    'r':[255,240,210,170,110],
    'g':[220,200,180,140,80],
    'b':[200,190,160,130,60],
    'skin_tone': ['fair','light','medium','olive','dark']
}
df = pd.DataFrame(data)

x=df[['r','g','b']]
y=df['skin_tone']

model = RandomForestClassifier()
model.fit(x,y)

with open("skin_tone_model.pkl","wb") as f:
          pickle.dump(model,f)

print(" Model trained to predict skin tone category(fair/dark/etc).")