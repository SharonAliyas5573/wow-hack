from django.shortcuts import render

def home(request):
    return render(request, 'home.html')
def service(request):
    return render(request, 'service.html')