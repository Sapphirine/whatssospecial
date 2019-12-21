from django.http import HttpResponse
from django.shortcuts import render
import pandas_gbq
import datetime
from google.oauth2 import service_account

input1

# Make sure you have installed pandas-gbq at first;
# You can use the other way to query BigQuery.
# please have a look at
# https://cloud.google.com/bigquery/docs/reference/libraries#client-libraries-install-nodejs
# To get your credential

credentials = service_account.Credentials.from_service_account_file('bighw3-5b739a0146d2.json')

def index(request):

    pandas_gbq.context.credentials = credentials
    pandas_gbq.context.project = "bighw3"

    data = {}
    context={}
    if:
        SQL = "SELECT special FROM `bighw3.Speciality.special` WHERE name = input1"
        list2 = pandas_gbq.read_gbq(SQL)
    else: 
        SQL = "SELECT Specialities FROM `bighw3.Speciality.special2` WHERE name = input1"
        list2 = pandas_gbq.read_gbq(SQL)
    
    context['content1'] = input1 
    context['data1']= list2
    
    return render(request, 'index.html', context)

def search(request):
    context={}
     if request.method == 'POST':
        search_id = request.POST.get('textfield', None)
     else:
         return render(request, 'search.html', context)

def homepage(request):
    context = {}
    return render(request, 'homepage.html', context)

