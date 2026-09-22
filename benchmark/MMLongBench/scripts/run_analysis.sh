#!/usr/bin/env bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BENCHMARK_ROOT=${BENCHMARK_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
PROJECT_ROOT=${PROJECT_ROOT:-$(cd -- "${BENCHMARK_ROOT}/../.." && pwd)}
MODEL_ROOT=${MODEL_ROOT:-${PROJECT_ROOT}/models}
RESULT_ROOT=${RESULT_ROOT:-${BENCHMARK_ROOT}/output}
TEST_FILE_ROOT=${TEST_FILE_ROOT:-${BENCHMARK_ROOT}/mmlb_data}
IMAGE_FILE_ROOT=${IMAGE_FILE_ROOT:-${BENCHMARK_ROOT}/mmlb_image}

cd "${BENCHMARK_ROOT}"

resolve_model_name() {
    local value="$1"
    case "${value}" in
        Qwen/Qwen2.5-VL-3B-Instruct|qwen2_5-vl-3b-instruct|qwen2.5-vl-3b-instruct)
            echo "${MODEL_ROOT}/qwen2.5-vl-3b-instruct"
            ;;
        Qwen/Qwen2.5-VL-7B-Instruct|qwen2_5-vl-7b-instruct|qwen2.5-vl-7b-instruct)
            echo "${MODEL_ROOT}/qwen2.5-vl-7b-instruct"
            ;;
        OpenGVLab/InternVL3-9B|internvl3-9b-instruct|internvl3-9b)
            echo "${MODEL_ROOT}/internvl3-9b"
            ;;
        *)
            echo "${value}"
            ;;
    esac
}

#--------------------------------OCR extracted DocVQA--------------------------------
model_name=Qwen/Qwen2.5-VL-3B-Instruct
dir_name=$(echo $model_name | rev | cut -d'/' -f1 | rev)
for task in "text_docqa"; do
    CUDA_VISIBLE_DEVICES=0 python eval.py --config configs/${task}_all.yaml --model_name_or_path "$(resolve_model_name "${model_name}")" \
    --output_dir ${RESULT_ROOT}/${dir_name} \
    --test_file_root ${TEST_FILE_ROOT} \
    --image_file_root ${IMAGE_FILE_ROOT} \
    --num_workers 16 --test_length 8,16,32,64,128
done

model_name=Qwen/Qwen2.5-3B-Instruct
dir_name=$(echo $model_name | rev | cut -d'/' -f1 | rev)
for task in "text_docqa"; do
    CUDA_VISIBLE_DEVICES=0 python eval.py --config configs/${task}_all.yaml --model_name_or_path "$(resolve_model_name "${model_name}")" \
    --output_dir ${RESULT_ROOT}/${dir_name} \
    --test_file_root ${TEST_FILE_ROOT} \
    --image_file_root ${IMAGE_FILE_ROOT} \
    --num_workers 16 --test_length 8,16,32,64,128
done


model_name=Qwen/Qwen2.5-VL-7B-Instruct
dir_name=$(echo $model_name | rev | cut -d'/' -f1 | rev)
for task in "text_docqa"; do
    CUDA_VISIBLE_DEVICES=1 python eval.py --config configs/${task}_all.yaml --model_name_or_path "$(resolve_model_name "${model_name}")" \
    --output_dir ${RESULT_ROOT}/${dir_name} \
    --test_file_root ${TEST_FILE_ROOT} \
    --image_file_root ${IMAGE_FILE_ROOT} \
    --num_workers 16 --test_length 8,16,32,64,128
done


model_name=Qwen/Qwen2.5-7B-Instruct
dir_name=$(echo $model_name | rev | cut -d'/' -f1 | rev)
for task in "text_docqa"; do
    CUDA_VISIBLE_DEVICES=2 python eval.py --config configs/${task}_all.yaml --model_name_or_path "$(resolve_model_name "${model_name}")" \
    --output_dir ${RESULT_ROOT}/${dir_name} \
    --test_file_root ${TEST_FILE_ROOT} \
    --image_file_root ${IMAGE_FILE_ROOT} \
    --num_workers 16 --test_length 8,16,32,64,128
done

model_name=Qwen/Qwen2.5-VL-32B-Instruct
dir_name=$(echo $model_name | rev | cut -d'/' -f1 | rev)
for task in "text_docqa"; do
    CUDA_VISIBLE_DEVICES=4,5,6,7 python eval.py --config configs/${task}_all.yaml --model_name_or_path "$(resolve_model_name "${model_name}")" \
    --output_dir ${RESULT_ROOT}/${dir_name} \
    --test_file_root ${TEST_FILE_ROOT} \
    --image_file_root ${IMAGE_FILE_ROOT} \
    --num_workers 16 --test_length 8,16,32,64,128
done

model_name=Qwen/Qwen2.5-32B-Instruct
dir_name=$(echo $model_name | rev | cut -d'/' -f1 | rev)
for task in "text_docqa"; do
    CUDA_VISIBLE_DEVICES=0,1 python eval.py --config configs/${task}_all.yaml --model_name_or_path "$(resolve_model_name "${model_name}")" \
    --output_dir ${RESULT_ROOT}/${dir_name} \
    --test_file_root ${TEST_FILE_ROOT} \
    --image_file_root ${IMAGE_FILE_ROOT} \
    --num_workers 16 --test_length 8,16,32,64,128
done


model_name=google/gemma-3-4b-it
dir_name=$(echo $model_name | rev | cut -d'/' -f1 | rev)
for task in "text_docqa"; do
    CUDA_VISIBLE_DEVICES=2,3 python eval.py --config configs/${task}_all.yaml --model_name_or_path "$(resolve_model_name "${model_name}")" \
    --output_dir ${RESULT_ROOT}/${dir_name} \
    --test_file_root ${TEST_FILE_ROOT} \
    --image_file_root ${IMAGE_FILE_ROOT} \
    --num_workers 16 --test_length 8,16,32,64,128
done


model_name=google/gemma-3-12b-it
dir_name=$(echo $model_name | rev | cut -d'/' -f1 | rev)
for task in "text_docqa"; do
    CUDA_VISIBLE_DEVICES=2,3 python eval.py --config configs/${task}_all.yaml --model_name_or_path "$(resolve_model_name "${model_name}")" \
    --output_dir ${RESULT_ROOT}/${dir_name} \
    --test_file_root ${TEST_FILE_ROOT} \
    --image_file_root ${IMAGE_FILE_ROOT} \
    --num_workers 16 --test_length 8,16,32,64,128
done


model_name=google/gemma-3-27b-it
dir_name=$(echo $model_name | rev | cut -d'/' -f1 | rev)
for task in "text_docqa"; do
    CUDA_VISIBLE_DEVICES=4,5,6,7 python eval.py --config configs/${task}_all.yaml --model_name_or_path "$(resolve_model_name "${model_name}")" \
    --output_dir ${RESULT_ROOT}/${dir_name} \
    --test_file_root ${TEST_FILE_ROOT} \
    --image_file_root ${IMAGE_FILE_ROOT} \
    --num_workers 16 --test_length 8,16,32,64,128
done


#--------------------------------VRAG with images replaced by its entity names--------------------------------
model_name=Qwen/Qwen2.5-VL-3B-Instruct
dir_name=$(echo $model_name | rev | cut -d'/' -f1 | rev)
for task in "text_rag"; do
    CUDA_VISIBLE_DEVICES=2 python eval.py --config configs/${task}_all.yaml --model_name_or_path "$(resolve_model_name "${model_name}")" \
    --output_dir ${RESULT_ROOT}/${dir_name} \
    --test_file_root ${TEST_FILE_ROOT} \
    --image_file_root ${IMAGE_FILE_ROOT} \
    --num_workers 16 --test_length 8,16,32,64,128
done

model_name=Qwen/Qwen2.5-3B-Instruct
dir_name=$(echo $model_name | rev | cut -d'/' -f1 | rev)
for task in "text_rag"; do
    CUDA_VISIBLE_DEVICES=3 python eval.py --config configs/${task}_all.yaml --model_name_or_path "$(resolve_model_name "${model_name}")" \
    --output_dir ${RESULT_ROOT}/${dir_name} \
    --test_file_root ${TEST_FILE_ROOT} \
    --image_file_root ${IMAGE_FILE_ROOT} \
    --num_workers 16 --test_length 8,16,32,64,128
done


model_name=Qwen/Qwen2.5-VL-7B-Instruct
dir_name=$(echo $model_name | rev | cut -d'/' -f1 | rev)
for task in "text_rag"; do
    CUDA_VISIBLE_DEVICES=4 python eval.py --config configs/${task}_all.yaml --model_name_or_path "$(resolve_model_name "${model_name}")" \
    --output_dir ${RESULT_ROOT}/${dir_name} \
    --test_file_root ${TEST_FILE_ROOT} \
    --image_file_root ${IMAGE_FILE_ROOT} \
    --num_workers 16 --test_length 8,16,32,64,128
done


model_name=Qwen/Qwen2.5-7B-Instruct
dir_name=$(echo $model_name | rev | cut -d'/' -f1 | rev)
for task in "text_rag"; do
    CUDA_VISIBLE_DEVICES=5 python eval.py --config configs/${task}_all.yaml --model_name_or_path "$(resolve_model_name "${model_name}")" \
    --output_dir ${RESULT_ROOT}/${dir_name} \
    --test_file_root ${TEST_FILE_ROOT} \
    --image_file_root ${IMAGE_FILE_ROOT} \
    --num_workers 16 --test_length 8,16,32,64,128
done

model_name=Qwen/Qwen2.5-VL-32B-Instruct
dir_name=$(echo $model_name | rev | cut -d'/' -f1 | rev)
for task in "text_rag"; do
    CUDA_VISIBLE_DEVICES=6,7 python eval.py --config configs/${task}_all.yaml --model_name_or_path "$(resolve_model_name "${model_name}")" \
    --output_dir ${RESULT_ROOT}/${dir_name} \
    --test_file_root ${TEST_FILE_ROOT} \
    --image_file_root ${IMAGE_FILE_ROOT} \
    --num_workers 16 --test_length 8,16,32,64,128
done

model_name=Qwen/Qwen2.5-32B-Instruct
dir_name=$(echo $model_name | rev | cut -d'/' -f1 | rev)
for task in "text_rag"; do
    CUDA_VISIBLE_DEVICES=0,1 python eval.py --config configs/${task}_all.yaml --model_name_or_path "$(resolve_model_name "${model_name}")" \
    --output_dir ${RESULT_ROOT}/${dir_name} \
    --test_file_root ${TEST_FILE_ROOT} \
    --image_file_root ${IMAGE_FILE_ROOT} \
    --num_workers 16 --test_length 8,16,32,64,128
done

model_name=google/gemma-3-4b-it
dir_name=$(echo $model_name | rev | cut -d'/' -f1 | rev)
for task in "text_rag"; do
    CUDA_VISIBLE_DEVICES=2,3 python eval.py --config configs/${task}_all.yaml --model_name_or_path "$(resolve_model_name "${model_name}")" \
    --output_dir ${RESULT_ROOT}/${dir_name} \
    --test_file_root ${TEST_FILE_ROOT} \
    --image_file_root ${IMAGE_FILE_ROOT} \
    --num_workers 16 --test_length 8,16,32,64,128
done


model_name=google/gemma-3-12b-it
dir_name=$(echo $model_name | rev | cut -d'/' -f1 | rev)
for task in "text_rag"; do
    CUDA_VISIBLE_DEVICES=2,3 python eval.py --config configs/${task}_all.yaml --model_name_or_path "$(resolve_model_name "${model_name}")" \
    --output_dir ${RESULT_ROOT}/${dir_name} \
    --test_file_root ${TEST_FILE_ROOT} \
    --image_file_root ${IMAGE_FILE_ROOT} \
    --num_workers 16 --test_length 8,16,32,64,128
done


model_name=google/gemma-3-27b-it
dir_name=$(echo $model_name | rev | cut -d'/' -f1 | rev)
for task in "text_rag"; do
    CUDA_VISIBLE_DEVICES=0,1 python eval.py --config configs/${task}_all.yaml --model_name_or_path "$(resolve_model_name "${model_name}")" \
    --output_dir ${RESULT_ROOT}/${dir_name} \
    --test_file_root ${TEST_FILE_ROOT} \
    --image_file_root ${IMAGE_FILE_ROOT} \
    --num_workers 16 --test_length 8,16,32,64,128
done
