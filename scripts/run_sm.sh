# for single modal inference

METHOD_NAMES=("IO" "CoT" "LLM_Debate" "MAD" "ChatEval" "DMAD")
DATASETS=("MMLU" "MMLU-Pro" "MATH" "GSM-Hard" "MedMCQA" "MedQA" "GPQA")
MODEL_NAME="llama3_1-local"

for METHOD in "${METHOD_NAMES[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    echo "Running $METHOD on $DATASET ..."
    python inference.py --method_name "$METHOD" --test_dataset_name "$DATASET" --model_name "$MODEL_NAME" --sequential
  done
done

echo "All inference jobs finished."