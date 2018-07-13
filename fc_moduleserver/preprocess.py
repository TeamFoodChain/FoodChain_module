from PIL import Image
import numpy as np
import os
import itertools as it
import random

#superclass_dict = {'견과류':0, '과일류':1, '유제품':2, '육류':3}
superclass_dict = ['견과류', '과일류', '유제품', '육류']
subclass_dict = {'아몬드': 0, '호두': 1, '바나나': 2, '사과': 3, '요거트': 4, '우유': 5, '돼지고기': 6, '소고기': 7}

def create_data2tensor(batch_size, im_path):
    data = []
    # read image
    im = Image.open(im_path)
    im_tmp = np.array(im)

    # if image is grey scaled, convert 1-dimension to 3-dimension
    if np.shape(im_tmp) != (224,224,3):
        im_tmp = im_tmp.reshape(-1,224,224)
        im_tmp = np.concatenate((im_tmp, im_tmp, im_tmp), axis = 0)
        im_tmp = im_tmp.transpose(1,2,0)
        print('check shape : %s' %np.shape(im_tmp))

    im_tmp = np.ndarray.tolist(im_tmp)

    # zip with fake label for feeding to the network
    im_data = [im_tmp, 0]

    for i in range(16):
        data.append(im_data)
    print('number of created data : %d' %len(data))

    # Do it again for feeding to the network (trained with batch data size of 32)
    im = Image.open('./fc_moduleserver/pork.png')
    # im = Image.open('./fc_moduleserver/test_image/walnut.jpg')
    im_tmp = np.array(im)
    if np.shape(im_tmp) != (224,224,3):
        im_tmp = im_tmp.reshape(-1, 224, 224)
        im_tmp = np.concatenate((im_tmp, im_tmp, im_tmp), axis = 0)
        im_tmp = im_tmp.transpose(1,2,0)

    im_tmp = np.ndarray.tolist(im_tmp)
    im_data = [im_tmp, 0]
    for j in range(16):
        data.append(im_data)
    print('number of created data : %d' %len(data))

    data_x = []
    data_y = []

    for k in data:
        data_x.append(k[0])
        data_y.append(k[1])
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    data_x = data_x.reshape(-1,3,224,224).astype(np.float32)
    data_x = data_x.transpose(0,2,3,1)
    print('check shape : {}' .format(np.shape(data_x)))

    label = np.zeros([batch_size, 8])

    # create one-hot vector of label
    for i in range(len(data_y)):
        y = data_y[i]
        label[i][y] = 1

    return data_x, label

def batch_returning(data, size, cycle=False, batch_fn=lambda x:x):
    batch = []
    if cycle is True:
        data = it.cycle(data)

    for item in data:
        batch.append(item)
        if len(batch) >= size:
            yield batch_fn(batch)
            batch = []

    if len(batch) > 0:
        yield batch_fn(batch)

def get_test_tensor(batch_size, img_path):
    x, y = create_data2tensor(batch_size, img_path)
    tensor_of_test_data = batch_returning(it.cycle(list(zip(x, y))),
                                          batch_size,
                                          cycle=True,
                                          batch_fn=lambda x:list(zip(*x)))
    return tensor_of_test_data

def image_resizing(img_path):
    im = Image.open(img_path)
    im = im.resize((224,224), Image.ANTIALIAS)
    im.save(img_path)