#! /bin/bash

# add w/ GPT given coordinates -> mask
CUDA_VISIBLE_DEVICES=4 python test.py --input_image 00137_FLUX.1-dev.png --prompt "add a fork above the plate" --task "text-guided" --bbox_coordinates "[250, 100, 750, 300]"

# add w/ human drawing mask
CUDA_VISIBLE_DEVICES=4 python test.py --input_image 00137_FLUX.1-dev.png --prompt "add a fork above the plate" --task "text-guided" --human_in_the_loop

# remove object w/ RES
CUDA_VISIBLE_DEVICES=4 python test.py --input_image flux-fill-dev.png --prompt "remove the right blue car" --task "object-removal"

# remove object w/ human drawing mask
CUDA_VISIBLE_DEVICES=4 python test.py --input_image flux-fill-dev.png --prompt "remove the right blue car" --task "object-removal" --human_in_the_loop

# remove object w/ pre-given mask
CUDA_VISIBLE_DEVICES=4 python test.py --input_image flux-fill-dev.png --prompt "remove the right blue car" --task "object-removal" --mask_image car_mask.png