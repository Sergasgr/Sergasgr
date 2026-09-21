## Sergio Graciá del Cisne

Final-year Computer Science student interested in Machine Learning and AI, focused on fine-tuning and alignment of LLMs. Alongside this, I work as an AI Engineer at Dialapplet, working hands-on with Python/FastAPI services, MLOps infrastructure, Agentic AI, and NLP model training pipelines.

## Projects

Three-part exploration of a specialized coding LLM's full lifecycle: align it, compress it, serve it fast.

- **[CodeAlign](https://github.com/Sergasgr/CodeAlign)** — Post-training Qwen2.5-Coder-7B-Instruct with SFT + DPO, using a composite reward (execution success + code complexity + lint score) instead of binary pass/fail, to avoid reward-hacking toward code that passes tests but is needlessly complex. Covers 8 languages spanning JetBrains' IDE lineup.
- **[CodeAlign-Runtime](https://github.com/Sergasgr/CodeAlign-Runtime)** — Low-latency C++/CUDA inference engine for Qwen2.5-Coder-0.5B: hand-written GEMV/GEMM kernels, INT4 quantization, Flash-Decoding attention, and a full transformer decode loop — every optimization measured against a theoretical bandwidth ceiling, not reported in a vacuum.
- **CodeAlign-Distillation** *(future)* — Knowledge distillation + QAT to compress CodeAlign's DPO checkpoint into a 5-14x smaller model, closing the loop for CodeAlign-Runtime.

## Open Source Contributions

**[peft](https://github.com/huggingface/peft)** — Added benchmark experiments for 5 PEFT methods (BEFT, HiRA, AdaMSS, UniLoRA, VeLoRA) to the FLUX.2-klein image-gen benchmark, evaluated against LoRA/OFT baselines with learning-rate tuning (Optuna) and ablation sweeps. Implemented ASA training-callback support in the image-gen benchmark's `run.py` (previously unimplemented) and extended target-module coverage to single-stream `to_out` layers that the shared config missed. Exploratory and superseded runs are published in the PEFT benchmark graveyard as negative results — including VeLoRA's, where the benchmark's default gradient checkpointing already addresses the activation-memory cost the method targets, a combination the paper never tested.

[Discussion #3522](https://github.com/huggingface/peft/discussions/3522) · [All PRs](https://github.com/huggingface/peft/pulls?q=is%3Apr+author%3ASergasgr) · [Hugging Face activity](https://huggingface.co/Sergasgr/activity/community)
