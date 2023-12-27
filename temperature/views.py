from django.shortcuts import render
from joblib import load
from temperature.Params_Temp import getingX
import random


value_list = [[2.00000000e+00, 3.19000000e+01, 2.16000000e+01, 5.22633972e+01,
       9.06047211e+01, 2.98506886e+01, 2.40350093e+01, 5.69188993e+00,
       5.19374478e+01, 2.25508198e-01, 2.51771373e-01, 1.59444059e-01,
       1.27727264e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 3.76046000e+01, 1.27032000e+02, 4.47624000e+01,
       5.14100000e-01, 5.86931250e+03]]

value_list[0][4]=value_list[0][4]+random.randint(0, 5)
value_list[0][6]=value_list[0][6]+random.randint(0, 5)
value_list[0][8]=value_list[0][8]+random.randint(0, 5)
value_list[0][3]=value_list[0][3]+random.randint(0, 5)
print(1333543)
def predictor(request):
    model = load('mongo-seed/Temperatur min model.joblib')
    # params = request.GET.get('predict')
    y_pred_min = model.predict(getingX(value_list))
    print('Min:',y_pred_min)
    return render(request, 'predict.html',{'prediction':y_pred_min})