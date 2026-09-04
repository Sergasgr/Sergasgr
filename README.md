Final-year Computer Science student interested in Machine Learning and AI, focused on fine-tuning and alignment of LLMs. Alongside this, I work as an AI Engineer at Dialapplet, working hands-on with Python/FastAPI services, MLOps infrastructure, Agentic AI, and NLP model training pipelines.

## Projects

Three-part exploration of a specialized coding LLM's full lifecycle: align it, compress it, serve it fast.

- **[CodeAlign](https://github.com/Sergasgr/CodeAlign)** — Post-training Qwen2.5-Coder-7B-Instruct with SFT + DPO, using a composite reward (execution success + code complexity + lint score) instead of binary pass/fail, to avoid reward-hacking toward code that passes tests but is needlessly complex. Covers 8 languages spanning JetBrains' IDE lineup.
- **[CodeAlign-Runtime](https://github.com/TU-USUARIO/CodeAlign-Runtime)** *(in development)* — Minimal C++/CUDA inference engine to serve the model with low latency: naive CUDA kernels → optimized kernels → INT4 quantization, with a final comparison against llama.cpp planned.
- **CodeAlign-Distillation** *(future)* — Knowledge distillation + QAT to compress CodeAlign's DPO checkpoint into a 5-14x smaller model, closing the loop for CodeAlign-Runtime.

## Open Source Contributions

**[peft](https://github.com/huggingface/peft)** — Benchmarked 3 new PEFT methods (BEFT, HiRA, AdaMSS) against LoRA/OFT baselines on the image-gen benchmark, wired up ASA training-callback support in the image-gen benchmark's script `run.py` (previously unimplemented), and fixed a target-modules matching bug along the way. 

[Discussion #3522](https://github.com/huggingface/peft/discussions/3522) · PRs: [#3607](https://github.com/huggingface/peft/pull/3607) [#3640](https://github.com/huggingface/peft/pull/3640) [#3641](https://github.com/huggingface/peft/pull/3641) [#3661](https://github.com/huggingface/peft/pull/3661) [#3663](https://github.com/huggingface/peft/pull/3663)
