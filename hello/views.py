from django.shortcuts import render
from django.http import HttpResponse

from .models import Greeting

# from .twitterlearning import learning

# Create your views here.
def index(request):
    # return HttpResponse('Hello from Python!')
    # learning()
    return render(request, "index.html")


def details(request):
    # return HttpResponse('Hello from Python!')
    return render(request, "details.html")


def sharestory(request):
    # return HttpResponse('Hello from Python!')
    return render(request, "sharestory.html")


def results(request):

    greeting = Greeting()
    greeting.save()

    greetings = Greeting.objects.all()

    return render(request, "results.html", {"greetings": greetings})
