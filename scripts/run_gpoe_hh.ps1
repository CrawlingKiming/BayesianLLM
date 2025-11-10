param(
    [string]$ModelName = "meta-llama/Llama-2-7b-hf",
    [string]$RefModelName = "meta-llama/Llama-2-7b-hf",
    [string]$DatasetPath = "data/hh_rlhf_gpoe_train.jsonl",
    [string]$OutputDir = "outputs/gpoe_llama_hh",
    [int]$MaxSamples = 20000
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path env:PYTHONPATH) -or ($env:PYTHONPATH -notlike "*$(Get-Location)*")) {
    $env:PYTHONPATH = "$(Get-Location)" + $(if ($env:PYTHONPATH) { ";" + $env:PYTHONPATH } else { "" })
}

Write-Host "Preparing HH-RLHF subset..."
python scripts/prepare_hh_rlhf.py --tokenizer $ModelName --output $DatasetPath --max_samples $MaxSamples

$adapterNames = @("expert_alignment", "expert_helpful", "expert_safe")

Write-Host "Training GPOE with $($adapterNames.Count) experts..."
python examples/train_gpoe.py `
    --model_name $ModelName `
    --ref_model_name $RefModelName `
    --train_path $DatasetPath `
    --output_dir $OutputDir `
    --num_experts $adapterNames.Count `
    --adapter_names $adapterNames `
    --per_device_train_batch_size 1 `
    --mc_samples 4 `
    --beta 1.0 `
    --lambda_offdiag 0.01 `
    --learning_rate 5e-5 `
    --max_steps 200 `
    --save_steps 100 `
    --logging_steps 10

Write-Host "Training complete. Artifacts in $OutputDir"

