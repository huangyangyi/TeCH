set -x
export INPUT_FILE=$1;
export EXP_DIR=$2;
export SUBJECT_NAME=$(basename $1 | cut -d"." -f1);
export REPLICATE_API_TOKEN=""; # your replicate token for BLIP API
export CUDA_HOME=/usr/local/cuda-11.6/ #/your/cuda/home/dir;
export PYOPENGL_PLATFORM=osmesa
export MESA_GL_VERSION_OVERRIDE=4.1
export PYTHONPATH=$PYTHONPATH:$(pwd);

# Step 1: Preprocess image, get SMPL-X & normal estimation
python utils/body_utils/preprocess.py --in_path ${INPUT_FILE} --out_dir ${EXP_DIR}

# Step 2: Get BLIP prompt and gender, you can also use your own prompt
python utils/get_prompt_blip.py --img-path ${EXP_DIR}/png/${SUBJECT_NAME}_crop.png --out-path ${EXP_DIR}/prompt.txt
# python core/get_prompt.py ${EXP_DIR}/png/${SUBJECT_NAME}_crop.png
export PROMPT=`cat ${EXP_DIR}/prompt.txt| cut -d'|' -f1`
export GENDER=`cat ${EXP_DIR}/prompt.txt| cut -d'|' -f2`

# Step 3: Finetune Dreambooth model (minimal GPU memory requirement: 2x32G)
rm -rf ${EXP_DIR}/ldm
python utils/ldm_utils/main.py -t --data_root ${EXP_DIR}/png/ --logdir ${EXP_DIR}/ldm/ --reg_data_root data/dreambooth_data/class_${GENDER}_images/ --bg_root data/dreambooth_data/bg_images/ --class_word ${GENDER} --no-test --gpus 0,1
# Convert Dreambooth model to diffusers format
python utils/ldm_utils/convert_ldm_to_diffusers.py --checkpoint_path ${EXP_DIR}/ldm/_v1-finetune_unfrozen/checkpoints/last.ckpt --original_config_file utils/ldm_utils/configs/stable-diffusion/v1-inference.yaml --scheduler_type ddim --image_size 512 --prediction_type epsilon --dump_path ${EXP_DIR}/sd_model
# [Optional] you can delete the original ldm exp dir to save disk memory
rm -rf ${EXP_DIR}/ldm

# Step 4: Run geometry stage (Run on a single GPU)
python core/main.py --config configs/tech_geometry.yaml --exp_dir $EXP_DIR --sub_name $SUBJECT_NAME
python utils/body_utils/postprocess.py --dir $EXP_DIR/obj --name $SUBJECT_NAME

# Step 5: Run texture stage (Run on a single GPU)
python core/main.py --config configs/tech_texture.yaml --exp_dir $EXP_DIR --sub_name $SUBJECT_NAME

# [Optional] export textured mesh with UV map, using atlas for UV unwraping.
python core/main.py --config configs/tech_texture_export.yaml --exp_dir $EXP_DIR --sub_name $SUBJECT_NAME --test