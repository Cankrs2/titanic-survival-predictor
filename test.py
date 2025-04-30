import joblib
model = joblib.load('titanic_model.pkl')
print(type(model))