# v1.1 2021/03/10 change l-3 level to woody and herb
import warnings


# those warnings are propogeted from implementation of oneformer, we don't take care of them here.
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="transformers.models.oneformer.image_processing_oneformer",
)
# original warning message:
# .../transformers/models/oneformer/image_processing_oneformer.py:427:
# futurewarning: the `reduce_labels` argument is deprecated and will be removed in v4.27. please use `do_reduce_labels` instead.

warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")
# original warning message:
# .../torch/functional.py:504:
# userwarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.
# (triggered internally at /opt/conda/conda-bld/pytorch_1682343995026/work/aten/src/aten/native/tensorshape.cpp:3483.)
# return _vf.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]

warnings.filterwarnings(
    "ignore", category=FutureWarning, module="transformers.utils.generic"
)
# original warning message:
# .../transformers/utils/generic.py:311: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.

warnings.filterwarnings(
    "ignore", category=FutureWarning, module="huggingface_hub.file_download"
)
# original message:
# .../huggingface_hub/file_download.py:1132: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.

warnings.filterwarnings(
    "ignore", category=FutureWarning, module="transformers.modeling_utils"
)
# Loading the model will automatically execute arbitrary code the OneFormer team writtten,
# please check OneFormer's repository if you have any question:
# https://huggingface.co/spaces/shi-labs/OneFormer/tree/main/oneformer
#
# original message:
# .../transformers/modeling_utils.py:488: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.


from functools import partial
from os import walk
from os.path import join as joinpath
from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor

import argparse
import atexit
import cv2
import numpy as np
import os
import sys
import time
import torch


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
shi-labs/oneformer_ade20k_swin_tiny
shi-labs/oneformer_ade20k_swin_large
shi-labs/oneformer_ade20k_dinat_large
"""
parser.add_argument(
    "-m",
    "--model",
    help=model_helpstring,
    choices=[
        "shi-labs/oneformer_ade20k_swin_tiny",
        "shi-labs/oneformer_ade20k_swin_large",
        "shi-labs/oneformer_ade20k_dinat_large",
    ],
    metavar="MODEL",
    default="shi-labs/oneformer_ade20k_swin_large",
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
- literal "auto": program will attempt to utilize GPU. If failed, use CPUs. (default)
- literal "cpu": use CPUs.
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

model_name = args.model  # pre-trainded model
threshold = args.threshold

# ctx: mxnet context (use CPUs or which GPU)
if args.device.strip('"') == "auto":
    ctx = "cuda" if torch.cuda.is_available() else "cpu"
elif args.device.strip('"') == "cpu":
    ctx = "cpu"
elif args.device.strip('"').isdigit():
    ctx = f"cuda:{args.device}"
else:
    print(f"{sys.argv[0]}:FATAL:Unrecognized device name {args.device}.", sys.stderr)
    sys.exit(1)


del model_helpstring, device_helpstring, args


# ------------------------------------------------------------------------
# Prepare I/O Targets
# ------------------------------------------------------------------------
# following code read files in subdir
fileList = []
for root, dirs, files in walk(img_path):
    for f in files:
        fullpath = joinpath(root, f)
        fileList.append(fullpath)

# select only jpg and png files in fileList
imgFileList = [name for name in fileList if name.lower().endswith((".jpg", ".png"))]
all_file_list_len = len(imgFileList)

if all_file_list_len == 0:
    print(
        f"{sys.argv[0]}:WARNING:No image file with extension "
        f"'.jpg' or '.png' found in {img_path}, abort.",
        sys.stderr,
    )
    sys.exit(1)

count_start = time.time()

out_folder = time.strftime("%Y_%m%d_%H%M%S_LADECO", time.localtime())

if not os.path.isdir(out_folder):
    os.mkdir(out_folder)
file_name_attribute = joinpath(
    os.path.dirname(__file__), "ladeco_v11.txt"
)  # label file

# generate csv head
with open(file_name_attribute, encoding="utf-8-sig") as h:
    lines = h.readlines()
    labels_attributetitle = ",".join([item.rstrip() for item in lines])

outpath = joinpath(out_folder, out_folder + ".csv")
outfile = open(outpath, "w", encoding="utf-8-sig")
atexit.register(outfile.close)

header = "fid" + "," + labels_attributetitle
outfile.write(header + "\n")

error_path = joinpath(out_folder, "error.txt")


# ------------------------------------------------------------------------
# Set Inference Sub-Routine
# ------------------------------------------------------------------------
processor = OneFormerProcessor.from_pretrained(model_name)
model = OneFormerForUniversalSegmentation.from_pretrained(model_name).to(ctx)


def transformer_inference(
    image, model, processor, task_inputs, return_tensors, device
) -> torch.Tensor:
    """A function wraps inference procedual of HuggingFace transformers OneFormer's
    semantic segmantation.

    returns HxW torch.Tensor where H is height of input image array, W is the width
    """
    sample = processor(
        images=image, task_inputs=task_inputs, return_tensors=return_tensors
    ).to(device)

    with torch.no_grad():
        outputs = model(**sample)

    masks: list[torch.Tensor] = processor.post_process_semantic_segmentation(outputs)
    mask = masks[0]  # batch size is 1

    return mask.type(torch.uint8)


# to use `segment`, simply `mask = segment(image)`
segment = partial(
    transformer_inference,
    model=model,
    processor=processor,
    task_inputs=["semantic"],
    return_tensors="pt",
    device=ctx,
)


# ------------------------------------------------------------------------
# Calculating and Saving Results
# ------------------------------------------------------------------------
# fmt: off
for idx, i_name in enumerate(imgFileList):
    print(f'{idx + 1}/{all_file_list_len}')

    try:
        raw_data = np.fromfile(i_name, dtype=np.uint8)
        img = cv2.imdecode(raw_data, 1)

        mask = segment(img).cpu().numpy()

        print('height =', mask.shape[0])
        print('width =', mask.shape[1])
        print('total pixels =', mask.size)

        # jarray: container of area ratios for ADE20k - SceneParsing150 categories
        # its elements are area ratio of certain scene element to whole picture
        jarray = np.zeros(150)
        for j in range(jarray.size):
            n_pixels = np.count_nonzero(mask == j)
            jarray[j] = round(n_pixels / mask.size, 3)

        # clean value that smaller than threshold eg. < 0.01
        la = np.where(jarray < threshold, 0, jarray)

        ################################################
        # compute various Levels of landscape elements #
        ################################################
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
        L3_furniture = (
            la[15] + la[19] + la[32] + la[69] + la[87] + la[88] + la[125]
            + la[138] + la[149] + la[132]
        )
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
        NFI = L1_nature / (L1_nature + L1_man_made)

        # prepare CSV table
        j2 = np.array(
            [
                L1_nature, L1_man_made, L2_landform, L2_vegetation, L2_water, L2_bio, L2_sky, L2_archi, L2_street,
                L3_hori_land, L3_vert_land, L3_woody_plant, L3_herb_plant, L3_flower, L3_hori_water, L3_vert_water, L3_human, L3_animal, L3_sky,
                L3_architecture, L3_archi_parts, L3_roadway, L3_furniture, L3_vehicle, L3_sign,
                L4_ground, la[29], la[46], la[91], la[52], la[16], la[68], la[34],
                la[4], la[72], la[17], la[9], la[66],
                la[21], la[26], la[60], la[109], la[128], la[104], la[113],
                la[12], la[126], la[2],
                L4_Building, la[79], la[84], la[48], la[0], la[8], la[14], la[78], L4_Canopy, la[121],
                la[6], la[11], la[59],
                la[15], la[19], la[32], la[69],la[87], la[88], la[125], la[138], la[149], la[132],
                la[20], la[80], la[83], la[102], la[116], la[127],
                la[43], la[100], la[136],
                others,
                NFI
            ]
        )

        # write percentage
        jar = ','.join(str(rate) for rate in j2.round(4))
        if ' ' in i_name:
            filename = '"' + i_name + '"'
        else:
            filename = i_name
        outfile.write(filename + ',' + jar + '\n')
    except KeyboardInterrupt:
        print('\nInterrupted by user.')
        print('bye~')
        exit()
    except Exception as exc:
        exc_name = exc.__class__.__name__
        exc_msg = exc.__str__()

        with open(error_path, 'a+', encoding="utf-8-sig") as errfile:
            errfile.write(i_name + ' | ' + exc_name + ':' + exc_msg + '\n')

        print('Error! ' + exc_name + ':' + exc_msg, file=sys.stderr)
        print('On image: ' + i_name, file=sys.stderr)

elapsed_time = time.time() - count_start
print('excecute time =', elapsed_time, 's')
