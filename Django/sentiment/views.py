from django.shortcuts import render
from .predict import getPrediction

# Create your views here.
def home(request):
    return render(request, 'home.html')

def result(request):
    data = str(request.GET['data'])

    result1, result2 = getPrediction(data)
    return render(request, 'result.html', {'result_bert_clf': result1, 'result_bert_reg': result2})