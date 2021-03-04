
import os, sys

# sys.path.append(os.path.dirname(__file__))
sys.path.append('/home/fabien/Kaggle/HuBMAP/HengCherKeng/2020-12-11/code/dummy_01/')


from common  import *
from hubmap_v2  import *


'''
          id                                           encoding
0  2f6ecfcdf  296084587 4 296115835 6 296115859 14 296147109...
1  aaa6a05cc  30989109 59 31007591 64 31026074 68 31044556 7...
2  cb2d976f4  78144363 5 78179297 15 78214231 25 78249165 35...
3  0486052bb  101676003 6 101701785 8 101727568 9 101753351 ...
4  e79de561c  7464094 14 7480273 41 7496453 67 7512632 82 75...
5  095bf7a1f  113430380 22 113468538 67 113506697 111 113544...
6  54f2eec69  124601765 36 124632133 109 124662536 147 12469...
7  1e2425f28  49453112 7 49479881 22 49506657 31 49533433 40...
(8, 2) 

'''
# <todo> make difference scale tile




#make tile train image
def run_make_train_tile():

    tile_scale = 0.25
    tile_min_score = 0.25
    tile_size = 320  #480 #
    tile_average_step = 160 #240  # 192


    train_tile_dir = '/root/share1/kaggle/2020/hubmap/data/hubmap-kidney-segmentation/etc/tile/0.25_320_160_train_corrected'
    #train_tile_dir = '/root/share1/kaggle/2020/hubmap/data/hubmap-kidney-segmentation/etc/tile/0.25_480_240_train_corrected'
    #train_tile_dir = '/root/share1/kaggle/2020/hubmap/data/hubmap-kidney-segmentation/etc/tile/0.25_320_240_train_corrected'
    #---

    df_train = pd.read_csv(data_dir + '/train.csv')
    print(df_train)
    print(df_train.shape)

    os.makedirs(train_tile_dir, exist_ok=True)
    for i in range(0,len(df_train)):
        id, encoding = df_train.iloc[i]

        image_file = data_dir + '/train/%s.tiff' % id
        image = read_tiff(image_file)

        height, width = image.shape[:2]
        #mask = rle_decode(encoding, height, width, 255)
        try:
            mask_file = data_dir + '/train/%s.corrected_shift_mask.png' % id
            mask = read_mask(mask_file)
        except:
            mask_file = data_dir + '/train/%s.corrected_mask.png' % id
            mask = read_mask(mask_file)

        structure = draw_strcuture_from_hue(image, fill=255, scale=tile_scale/32)
        print(id, mask_file)

        #make tile
        tile = to_tile(image, mask, structure, tile_scale, tile_size, tile_average_step, tile_min_score)

        coord = np.array(tile['coord'])
        df_image = pd.DataFrame()
        df_image['cx']=coord[:,0].astype(np.int32)
        df_image['cy']=coord[:,1].astype(np.int32)
        df_image['cv']=coord[:,2]

        # --- save ---
        os.makedirs(train_tile_dir+'/%s'%id, exist_ok=True)

        tile_id =[]
        num = len(tile['tile_image'])
        for t in range(num):
            cx,cy,cv   = tile['coord'][t]
            s = 'y%08d_x%08d' % (cy, cx)
            tile_id.append(s)

            tile_image = tile['tile_image'][t]
            tile_mask  = tile['tile_mask'][t]
            cv2.imwrite(train_tile_dir + '/%s/%s.png' % (id, s), tile_image)
            cv2.imwrite(train_tile_dir + '/%s/%s.mask.png' % (id, s), tile_mask)

            image_show('tile_image', tile_image)
            image_show('tile_mask', tile_mask)
            cv2.waitKey(1)


        df_image['tile_id']=tile_id
        df_image[['tile_id','cx','cy','cv']].to_csv(train_tile_dir+'/%s.csv'%id, index=False)
        #------



#make tile train image
def run_make_test_tile():

    tile_scale = 0.25
    tile_min_score = 0.25
    tile_size = 320  # 480 #
    tile_average_step = 160 #240  # 160 #192


    test_tile_dir = '/root/share1/kaggle/2020/hubmap/data/hubmap-kidney-segmentation/etc/tile/0.25_320_160_test'
    #---


    os.makedirs(test_tile_dir, exist_ok=True)
    for id in [
        'c68fe75ea',
        'afa5e8098',
    ]:
        print(id)

        image_file = data_dir + '/test/%s.tiff' % id
        json_file  = data_dir + '/test/%s-anatomical-structure.json' % id

        image = read_tiff(image_file)
        height, width = image.shape[:2]

        mask = None
        structure = draw_strcuture(read_json_as_df(json_file), height, width, structure=['Cortex'])
        # structure = draw_strcuture_from_hue(image, fill=255, scale=tile_scale/32)

        #make tile
        tile = to_tile(image, mask, structure, tile_scale, tile_size, tile_average_step, tile_min_score)

        coord = np.array(tile['coord'])
        df_image = pd.DataFrame()
        df_image['cx']=coord[:,0].astype(np.int32)
        df_image['cy']=coord[:,1].astype(np.int32)
        df_image['cv']=coord[:,2]

        # --- save ---
        os.makedirs(test_tile_dir+'/%s'%id, exist_ok=True)

        tile_id =[]
        num = len(tile['tile_image'])
        for t in range(num):
            cx,cy,cv   = tile['coord'][t]
            s = 'y%08d_x%08d' % (cy, cx)
            tile_id.append(s)

            tile_image = tile['tile_image'][t]
            cv2.imwrite(test_tile_dir + '/%s/%s.png' % (id, s), tile_image)
            image_show('tile_image', tile_image)
            cv2.waitKey(1)


        df_image['tile_id']=tile_id
        df_image[['tile_id','cx','cy','cv']].to_csv(test_tile_dir+'/%s.csv'%id, index=False)
        #------

# run_make_test_tile()
# exit(0)
#
#



#make tile train image
def run_make_train_mask():

    df_train = pd.read_csv(data_dir + '/train.csv')
    print(df_train)
    print(df_train.shape)

    for i in range(0,len(df_train)):
        id, encoding = df_train.iloc[i]

        image_file = data_dir + '/train/%s.tiff' % id
        image = read_tiff(image_file)

        height, width = image.shape[:2]
        mask = rle_decode(encoding, height, width, 255)

        cv2.imwrite(data_dir + '/train/%s.mask.png' % id, mask)


def run_make_train_sample_overlay():
    train_tile_dir = '/root/share1/kaggle/2020/hubmap/data/hubmap-kidney-segmentation/tile/0.25_480_240_train'

    df_train = pd.read_csv(data_dir + '/train.csv')
    print(df_train)
    print(df_train.shape)

    scale = 0.25
    for i in range(0,len(df_train)):
        id, encoding = df_train.iloc[i]

        image_file = data_dir + '/train/%s.tiff' % id
        image = read_tiff(image_file)

        image_small = cv2.resize(image, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        df = pd.read_csv(train_tile_dir + '/%s.csv'% (id) )
        coord = df[['cx','cy','cv']].values

        overlay = image_small.copy()
        for cx,cy,cv in coord:
            cx = int(cx)
            cy = int(cy)
            cv = int(255 * cv)
            cv2.circle(overlay, (cx, cy), 64, [cv,cv,cv], -1)
            cv2.circle(overlay, (cx, cy), 64, [0, 0, 255],16)


        overlay = cv2.resize(overlay, dsize=None, fx=0.1, fy=0.1, interpolation=cv2.INTER_LINEAR)
        image_show('overlay', overlay)
        cv2.waitKey(0)

        cv2.imwrite(train_tile_dir + '/%s.sample.png'% (id), overlay)





#make tile train image
def run_make_pseudo_tile():

    pseudo_tile_dir = '/root/share1/kaggle/2020/hubmap/data/hubmap-kidney-segmentation/tile/0.25_480_240_pseudo'
    #---

    #df_train = pd.read_csv(data_dir + '/train.csv')
    df_pseudo = pd.read_csv('/root/share1/kaggle/2020/hubmap/result/resnet34/fold2/submit-fold-2-resnet34-00010000_model_lb0.837.csv')
    print(df_pseudo)
    print(df_pseudo.shape)

    os.makedirs(pseudo_tile_dir, exist_ok=True)
    for i in range(0,len(df_pseudo)):
        id, encoding = df_pseudo.iloc[i]

        image_file = data_dir + '/test/%s.tiff' % id
        image = read_tiff(image_file)

        height, width = image.shape[:2]
        mask = rle_decode(encoding, height, width, 255)

        #make tile
        tile = to_tile(image, mask, tile_scale, tile_size, tile_average_step, tile_min_score)
        #mask, scale, size, step, min_score

        coord = np.array(tile['coord'])
        df_image = pd.DataFrame()
        df_image['cx']=coord[:,0].astype(np.int32)
        df_image['cy']=coord[:,1].astype(np.int32)
        df_image['cv']=coord[:,2]

        # --- save ---
        os.makedirs(pseudo_tile_dir + '/%s'%id, exist_ok=True)

        tile_id =[]
        num = len(tile['tile_image'])
        for t in range(num):
            cx,cy,cv   = tile['coord'][t]
            s = 'y%08d_x%08d' % (cy, cx)
            tile_id.append(s)

            tile_image = tile['tile_image'][t]
            tile_mask  = tile['tile_mask'][t]
            cv2.imwrite(pseudo_tile_dir + '/%s/%s.png' % (id, s), tile_image)
            cv2.imwrite(pseudo_tile_dir + '/%s/%s.mask.png' % (id, s), tile_mask)

            image_show('tile_image', tile_image)
            image_show('tile_mask', tile_mask)
            cv2.waitKey(1)


        df_image['tile_id']=tile_id
        df_image[['tile_id','cx','cy','cv']].to_csv(pseudo_tile_dir+'/%s.csv'%id, index=False)
        #------


#################################################################



def run_check_tile():

    #load a train image
    id = 'e79de561c'
    image_file = data_dir + '/train/%s.tiff' % id
    image = read_tiff(image_file)
    height, width = image.shape[:2]

    #load a mask
    df = pd.read_csv(data_dir + '/train.csv', index_col='id')
    encoding = df.loc[id,'encoding']
    mask = rle_decode(encoding, height, width, 255)

    #make tile
    tile = to_tile(image, mask)


    if 1: #debug
        overlay = tile['image_small'].copy()
        for cx,cy,cv in tile['coord']:
            cv = int(255 * cv)
            cv2.circle(overlay, (cx, cy), 64, [cv,cv,cv], -1)
            cv2.circle(overlay, (cx, cy), 64, [0, 0, 255], 16)
        for cx,cy,cv in tile['reject']:
            cv = int(255 * cv)
            cv2.circle(overlay, (cx, cy), 64, [cv,cv,cv], -1)
            cv2.circle(overlay, (cx, cy), 64, [255, 0, 0], 16)

        #---
        num = len(tile['coord'])
        cx, cy, cv = tile['coord'][num//2]
        cv2.rectangle(overlay,(cx-tile_size//2,cy-tile_size//2),(cx+tile_size//2,cy+tile_size//2), (0,0,255), 16)

        image_show('overlay', overlay, resize=0.1)
        cv2.waitKey(1)

    # make prediction for tile
    # e.g. predict = model(tile['tile_image'])
    tile_predict = tile['tile_mask'] # dummy: set predict as ground truth

    # make mask from tile
    height, width = tile['image_small'].shape[:2]
    predict = to_mask(tile_predict, tile['coord'],  height, width)

    truth = tile['mask_small']#.astype(np.float32)/255
    diff = np.abs(truth-predict)
    print('diff', diff.max(), diff.mean())

    if 1:
        image_show_norm('diff', diff, min=0, max=1, resize=0.2)
        image_show_norm('predict', predict, min=0, max=1, resize=0.2)
        cv2.waitKey(0)


# main #################################################################
if __name__ == '__main__':
    run_check_tile()
    # run_make_train_tile()
    #run_make_train_mask()
    #run_make_train_sample_overlay()

    #run_make_pseudo_tile()