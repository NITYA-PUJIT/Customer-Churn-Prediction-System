import pickle
import numpy as np

with open("model/churn_model.pkl", "rb") as f:
    model = pickle.load(f)

sample = np.array([
    1,0,1,0,12,1,0,1,0,0,0,0,1,1,1,0,1,70.35,845.5
]).reshape(1, -1)

prediction = model.predict(sample)

print("Churn" if prediction[0] == 1 else "No Churn")
