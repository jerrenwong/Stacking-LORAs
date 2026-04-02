LoRA Reversal Experiment
Research Question
When you train a LoRA adapter to learn behavior X, and then want to undo X, is it easier to continue training the same LoRA adapter (condition ii) versus merging and training a new LoRA adapter (condition i)?
Experiment Design
Setup

Base model: An English-dominant instruct model (e.g., Qwen/Qwen2.5-1.5B-Instruct for fast iteration, scale up to 7B+ later)
Feature X: Respond in Chinese instead of English
Anti-X: Respond in English (i.e., restore original behavior)

Phases
Phase 1 — Train X (English → Chinese):
LoRA fine-tune the base model on English prompts paired with Chinese responses. The model should learn to respond in Chinese.
Phase 2 — Train anti-X (Chinese → English), two conditions:

Condition (i) — Merge then new LoRA: Merge Phase 1 LoRA into base weights → attach a freshly initialized LoRA (same config) → SFT on English prompt-response pairs
Condition (ii) — Continue same LoRA: Keep Phase 1 LoRA unmerged → continue training the same LoRA on English prompt-response pairs

Both conditions use identical training data, hyperparameters, and number of steps. The only difference is whether the LoRA adapter is fresh or continued.
Datasets
Use real instruction-following datasets. Suggested source:

Prompts: Use the prompts/instructions from yahma/alpaca-cleaned (52k instructions) or Open-Orca/OpenOrca or any similar instruction dataset from HuggingFace.
Chinese responses: Generate Chinese responses by either:

Using a bilingual model to translate the English responses, OR
Simpler: just take the English prompts and pair them with Chinese responses from a Chinese instruction dataset like silk-road/alpaca-data-gpt4-chinese or shibing624/alpaca-zh


English responses: Use the original English responses from the same dataset.
Split: Use ~5k examples for training, 500 held-out for eval. Keep train/eval splits consistent across all conditions.

If translation is too complex for the prototype, a simpler approach:

Take N prompts from yahma/alpaca-cleaned
For Chinese SFT data: prepend a system prompt like "Always respond in Chinese" and generate Chinese responses using the base model or any bilingual model, then use those as targets
For English SFT data: use the original English responses from Alpaca

Evaluation
Primary metric: Language ratio — for each eval prompt, generate a response and classify its language using langdetect or fasttext language ID. Report % English and % Chinese.
Secondary metrics:

Response quality: perplexity on a held-out English QA set, or a simple downstream benchmark
Distance to original: KL divergence between the restored model and the original base model on the eval set

Eval schedule: Evaluate every N training steps during Phase 2 to produce convergence curves.
Variables

Primary comparison: Condition (i) vs Condition (ii), measured by convergence speed (steps to reach X% English) and final quality
Rank sweep: Run the full experiment at LoRA ranks [4, 8, 16, 32, 64] to see if rank affects the advantage
Control: always use the same lora_alpha = 2 * rank, same target modules, same learning rate, same batch size

Project Structure
lora-reversal/
├── CLAUDE.md              # This file
├── run_experiment.py       # Main experiment script
├── data.py                 # Dataset loading and formatting
├── eval.py                 # Language detection and evaluation
├── plot_results.py         # Plot convergence curves
├── requirements.txt        # Dependencies
└── results/                # Output directory
    ├── phase1_eval.json
    ├── phase2_continue_eval.json
    ├── phase2_new_eval.json
    ├── summary.json
    └── figures/
        ├── convergence.png
        └── rank_sweep.png
Implementation Notes
run_experiment.py
Core experiment loop. Takes args for model name, rank, epochs, etc.
Steps:

Load base model and tokenizer
Evaluate base model language ratio (sanity check — should be high English)
Load training and eval datasets via data.py
Phase 1: Apply LoRA, train on Chinese data, save adapter. Evaluate to confirm model now responds in Chinese.
Phase 2 Condition (ii): Load saved Phase 1 adapter (unmerged, trainable). Train on English data. Log eval metrics every N steps.
Phase 2 Condition (i): Load saved Phase 1 adapter, merge into base. Apply fresh LoRA (same config). Train on English data. Log eval metrics every N steps.
Save all results to JSON.

Use argparse for CLI:
python run_experiment.py --model Qwen/Qwen2.5-1.5B-Instruct --rank 8
python run_experiment.py --model Qwen/Qwen2.5-1.5B-Instruct --sweep --ranks 4 8 16 32
data.py

load_training_data(tokenizer, n_train=5000, n_eval=500) → returns chinese_train, english_train, eval_prompts
Pull prompts from yahma/alpaca-cleaned via HuggingFace datasets
For Chinese targets: use silk-road/alpaca-data-gpt4-chinese or translate. If neither is practical, generate synthetic Chinese responses by prompting a bilingual model.
For English targets: use the original Alpaca English outputs
Format using the tokenizer's chat template
Tokenize with padding and label masking (mask padding tokens with -100, and ideally mask the prompt portion of labels so loss is only on the response)

eval.py

detect_language(text) -> str: Use langdetect or fasttext (lid.176.bin model). fasttext is more reliable for short texts.
evaluate_model(model, tokenizer, eval_prompts) -> dict: Generate responses with do_sample=False, classify language, return {"en_ratio": float, "zh_ratio": float, "responses": [...]}
Implement as a TrainerCallback that runs every N steps during training

plot_results.py

Load JSON eval histories from results/
Plot convergence curves: x-axis = training steps, y-axis = English ratio, two lines (condition i vs ii)
Plot rank sweep: x-axis = rank, y-axis = steps-to-90%-English (or similar threshold)
Save to results/figures/

requirements.txt
torch>=2.0
transformers>=4.40
peft>=0.10
datasets
accelerate
langdetect
fasttext
matplotlib
LoRA Config
Use the same config for all conditions and phases:
pythonLoraConfig(
    task_type="CAUSAL_LM",
    r=RANK,                  # variable
    lora_alpha=RANK * 2,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
Target all linear layers (attention + MLP) per best practices from Biderman et al. and Schulman et al.
Training Config
pythonTrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=5,
)
Keep epochs/steps identical between conditions. Start with 5-10 epochs for the prototype, adjust based on convergence.
Key Concerns

Phase 1 must actually work. Verify the model responds in Chinese after Phase 1 before proceeding. If the Chinese ratio isn't high (>80%), increase Phase 1 training.
Identical compute budget. Both conditions must train for exactly the same number of gradient steps in Phase 2.
Memory. copy.deepcopy of the base model is used to ensure conditions are independent. This doubles GPU memory. If memory is tight, save/reload from disk instead.
Reproducibility. Set seeds. Use do_sample=False for eval generation.
Label masking. Only compute loss on the response tokens, not the prompt. This is important for clean training signal.
