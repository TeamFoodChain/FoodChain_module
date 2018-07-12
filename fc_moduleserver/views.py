from django.shortcuts import render
from fc_moduleserver import module_executor
from django.core.files.images import ImageFile
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
import os

# Create your views here.
@api_view(['GET','POST'])
def classify_image(request):
    if request.method == 'POST':
        print('1')
        return Response('1111')
    else :
        print(os.getcwd())
        input_file = os.listdir('./fc_moduleserver/test_image')
        print('list of file : %s' %input_file)
        answer = module_executor.test('./fc_moduleserver/test_image/'+input_file[0])
        print('answer : %s' %answer)
        return Response(answer)