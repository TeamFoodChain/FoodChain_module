from fc_moduleserver import module_executor
from rest_framework.decorators import api_view
from rest_framework.response import Response
import requests
from fc_moduleserver import preprocess

# Create your views here.
@api_view(['GET','POST'])
def classify_image(request):
    if request.method == 'POST':
        print('1')
        return Response('1111')

    else :
        #url = request.data
        url = 'https://s3.ap-northeast-2.amazonaws.com/foodchainimage/1531425656582.jpg'
        test_image = './fc_moduleserver/test_image/test_image.jpg'
        image_data = requests.get(url, stream=True)
        with open(test_image, 'wb') as handler:
            handler.write(image_data.content)

        preprocess.image_resizing(test_image)
        answer, sup = module_executor.test(test_image)

        print('answer : %s' %answer)
        return Response([sup, answer])
        #return Response('1')
