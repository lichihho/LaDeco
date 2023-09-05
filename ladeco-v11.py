# v1.1 2021/03/10 change l-3 level to woody and herb

import argparse
import os
import sys
from os import walk
from os.path import join

# following statement is to fasten processing speed, also prevent crash
os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"

import time

import cv2
import gluoncv
import mxnet as mx
import numpy as np
from gluoncv.data.transforms.presets.segmentation import test_transform


def read_image(image):
    raw_data = np.fromfile(image, dtype=np.uint8)
    img = cv2.imdecode(raw_data, 1)
    return img


# ------------------------------------------------------------------------
# Parse Arguments
# ------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description=(
        "Calculate the proportion of landscape elements in each image within the folder, "
        "with each image being computed independently."
    ),
)
parser.add_argument("source", metavar="IMG_FOLDER", help="path to image folder.")

model_helpstring = """\
The semantic segmantation model.
Choose one from below:
fcn_resnet50_ade
fcn_resnet101_ade
psp_resnet50_ade
psp_resnet101_ade
deeplab_resnet50_ade
deeplab_resnet101_ade
deeplab_resnest50_ade
deeplab_resnest101_ade
deeplab_resnest200_ade
deeplab_resnest269_ade (default)
"""
parser.add_argument(
    "-m",
    "--model",
    help=model_helpstring,
    default="deeplab_resnest269_ade",
    choices=[
        "fcn_resnet50_ade",
        "fcn_resnet101_ade",
        "psp_resnet50_ade",
        "psp_resnet101_ade",
        "deeplab_resnet50_ade",
        "deeplab_resnet101_ade",
        "deeplab_resnest50_ade",
        "deeplab_resnest101_ade",
        "deeplab_resnest200_ade",
        "deeplab_resnest269_ade",
    ],
    metavar="MODEL",
)
parser.add_argument(
    "-t",
    "--threshold",
    help="The thresold to round an element to zero. (default 0.01)",
    type=float,
    default=0.01,
)

device_helpstring = """\
Use CPU or GPU.
Choice are:
- literal "auto": program will attempt to utilize GPU. If failed, use CPU. (default)
- literal "cpu": use CPU.
- literal "gpu": use the first GPU device.
- a digit like 0, 1, or 2: Use the GPU device specified by the index.
"""
parser.add_argument("-d", "--device", help=device_helpstring, default="auto")

args = parser.parse_args()


img_path = args.source  # path to image folder

if not os.path.exists(img_path):
    print(f"{sys.argv[0]}:FATAL:{img_path} not exists.", sys.stderr)
    sys.exit(1)
if not os.path.isdir(img_path):
    print(f"{sys.argv[0]}:FATAL:{img_path} is not a folder.", sys.stderr)
    sys.exit(1)

model_name = "deeplab_resnest269_ade"  # pre-trainded model
threshold = args.threshold

# ctx: mxnet context (use CPUs or which GPU)
if args.device.strip('"') == "auto":
    ctx = mx.gpu() if mx.context.num_gpus() else mx.cpu()
elif args.device.strip('"') == "cpu":
    ctx = mx.cpu()
elif args.device.strip('"') == "gpu":
    ctx = mx.gpu()
elif args.device.strip('"').isdigit():
    ctx = mx.gpu(int(args.device.strip('"')))
else:
    print(f"{sys.argv[0]}:FATAL:Unrecognized device name {args.device}.", sys.stderr)
    sys.exit(1)


del model_helpstring, device_helpstring, args


# ------------------------------------------------------------------------
#
# ------------------------------------------------------------------------
# following code read files in subdir
fileList = []
for root, dirs, files in walk(img_path):
    for f in files:
        fullpath = join(root, f)
        fileList.append(fullpath)
        # print(fullpath)

# select only jpg and png files in fileList
imgFileList = [name for name in fileList if name.lower().endswith((".jpg", ".png"))]
all_file_list_len = len(imgFileList)
# imgFileList = sorted(glob.glob(img_path + '*.[Jj][Pp][Gg]') + glob.glob(img_path + '*.[Pp][Nn][Gg]'))


csv_name = model_name + "_sceneElements.csv"
count_start = time.time()

camFolderName = time.strftime("%Y_%m%d_%H%M%S_LADECO", time.localtime())
# imgfoldername = os.getcwd() +'\\' + camFolderName + '\\'+ camFolderName +'_img' #segmentaion image folder

if not os.path.isdir(camFolderName):
    os.mkdir(camFolderName)
    # os.mkdir(imgfoldername)
file_name_attribute = "ladeco_v11.txt"  # label file

# generate csv head
with open(file_name_attribute, encoding="utf-8") as h:
    lines = h.readlines()
    labels_attributetitle = ",".join([item.rstrip() for item in lines])


g = open(camFolderName + "\\" + camFolderName + ".csv", "a+")
g.write("fid," + labels_attributetitle + "\n")

model = gluoncv.model_zoo.get_model(
    model_name, pretrained=True, ctx=mx.gpu(0)
)  # get model

error_txt_path = os.getcwd() + "\\" + camFolderName + "\\error.txt"  # save error txt

# ------------------------------------------------------------------------
#
# ------------------------------------------------------------------------
# fmt: off
count = 0
for i_name in imgFileList:
    try:
        count += 1
        print(f'{count}/{all_file_list_len}')
        #now_file_name = i_name.replace(img_path,'')[0:22]
        #now_file_name = i_name.replace(img_path,'')
        #index_num = imgFileList.index(i_name)
        
        #img = image.imread(i_name)
        # read chinese path
        img = read_image(i_name)
        img = mx.nd.array(img)
    
        #segmentation
        img = test_transform(img, ctx)
        output = model.predict(img)
        predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy() 
    
        print('height = ',len(predict))
        print('width = ',len(list(predict[0])))
    
    
        #compute all pixels
        a=predict.shape
        allpix=a[0]*a[1]
        jarray=np.zeros([150])
        #j2 = np.zeros([74])
        print(allpix)
    
        for j in range(150):
            o=round( (np.count_nonzero(predict == j))/allpix, 3)
            jarray[j]=o
            
        # clean value that smaller than threshold eg. < 0.01
        jarray=np.where(jarray < threshold, 0, jarray);
        
        
        # compute various Levels of landscape elements
        la = jarray
        
        # Merge L-4 element
        L4_ground = la[13] + la[94] # Earth/Ground + Ground/Land
        L4_Building = la[1] + la[25] # Building + House
        L4_Canopy = la[86] + la[106] # Sunshade + Canopy
        
        # Compute L-3 Level
        #Landform
        L3_hori_land = L4_ground + la[29] + la[46] + la[91] + la[52]
        L3_vert_land = la[16] + la[68] + la[34]
        
        #Vegetation
        L3_woody_plant = la[4] + la[72] + la[17]
        L3_herb_plant = la[9]
        L3_flower = la[66]
        
        #Water
        L3_hori_water = la[21] + la[26] + la[60] + la[109] + la[128]
        L3_vert_water = la[104] + la[113]
        
        #Bio
        L3_human = la[12]
        L3_animal = la[126]
        
        #Sky
        L3_sky = la[2]
        
        #Archi
        L3_architecture = L4_Building + la[79] + la[84] + la[48]
        L3_archi_parts = la[0] +la[8] + la[14] + la[78] + L4_Canopy + la[121]
        
        #Street
        L3_roadway = la[6] + la[11] + la[52] + la[59]
        L3_furniture = la[15] + la[19] + la[32] + la[69] + la[87] + la[88] +la[125] +\
            la[138] + la[149] + la[132]
        L3_vehicle = la[20] + la[80] + la[83] + la[102] + la[116] + la[127]
        L3_sign = la[43] + la[100] + la[136]
        
        # Compute L-2 Level
        L2_landform = L3_hori_land + L3_vert_land
        L2_vegetation = L3_woody_plant + L3_herb_plant + L3_flower
        L2_water = L3_hori_water + L3_vert_water
        L2_bio = L3_human + L3_animal
        L2_sky = L3_sky
        L2_archi = L3_architecture + L3_archi_parts
        L2_street = L3_roadway + L3_furniture + L3_vehicle + L3_sign
        
        # Compute L-1 Level
        L1_nature = L2_landform + L2_vegetation + L2_water + L2_bio + L2_sky
        L1_man_made = L2_archi + L2_street
        
        # others
        others = 1 - L1_nature - L1_man_made


        # Landscape Character
        NFI = (L1_nature) / (L1_nature + L1_man_made) 
                    # correlation with 480img was 0.896
        
        # prepare CSV table
        j2 = np.array([L1_nature, L1_man_made, L2_landform, L2_vegetation, L2_water, L2_bio, L2_sky, L2_archi, L2_street,\
                L3_hori_land, L3_vert_land, L3_woody_plant, L3_herb_plant, L3_flower, L3_hori_water, L3_vert_water, L3_human, L3_animal, L3_sky,\
                L3_architecture, L3_archi_parts, L3_roadway, L3_furniture, L3_vehicle, L3_sign,\
                L4_ground, la[29], la[46], la[91], la[52], la[16], la[68], la[34],\
                la[4], la[72], la[17], la[9], la[66],\
                la[21], la[26], la[60], la[109], la[128], la[104], la[113],\
                la[12], la[126], la[2],\
                L4_Building, la[79], la[84], la[48], la[0], la[8], la[14], la[78], L4_Canopy, la[121],\
                la[6], la[11], la[59],\
                la[15], la[19], la[32], la[69],la[87], la[88], la[125], la[138], la[149], la[132],\
                la[20], la[80], la[83], la[102], la[116], la[127],\
                la[43], la[100], la[136],\
                others,\
                NFI])
            
    
        # write percentage
        jar=np.array2string(j2, separator=',')[1:-1] 
        jar = jar.replace('\n','') 
        #print(jar)
        g.write(i_name + ',' + jar + '\n')
    
    

    except:
        with open(error_txt_path, 'a+') as txt:
            txt.writelines(f'{i_name}' + '\n')
        print('Something Error!')
g.close()

count_end = time.time()
print('excecute time = ',count_end - count_start,'s')
