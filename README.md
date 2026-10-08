## Sergio Graciá del Cisne

Final-year Computer Science student interested in Machine Learning and AI, focused on fine-tuning and alignment of LLMs. Alongside this, I work as an AI Engineer at Dialapplet, working hands-on with Python/FastAPI services, MLOps infrastructure, Agentic AI, and NLP model training pipelines.

## Projects

Three-part exploration of a specialized coding LLM's full lifecycle: align it, compress it, serve it fast.

CodeAlign (align) → CodeAlign-Distill (compress) → CodeAlign-Runtime (serve)

- **[CodeAlign](https://github.com/Sergasgr/CodeAlign)** — End-to-end post-training of Qwen2.5-Coder-7B-Instruct on one 16 GB GPU: 122K CommitPackFT samples curated across 8 languages, QLoRA SFT, preference pairs labelled by a Docker sandbox (Rust gRPC daemon) and static analysis, and DPO with a composite vs. execution-only reward plus a size-matched control. The evaluation reproduces the base model's published HumanEval score (90.2%); the ablations show both offline rewards being gamed (commented-out code "runs without errors") and trace v1.0's DPO collapse to over-optimization. v2 turns it into a benchmark of PEFT, alignment and distillation methods for code.
- **CodeAlign-Distill *(future)*** — Compresses a 7B code model into a 0.5B one that CodeAlign-Runtime serves in INT4: knowledge distillation from Qwen2.5-Coder-7B-Instruct, then hand-written GPTQ and QAT (STE, LSQ) in the engine's exact weight format, evaluated through the engine itself.
- **[CodeAlign-Runtime](https://github.com/Sergasgr/CodeAlign-Runtime)** — C++/CUDA inference engine for Qwen2.5-Coder-0.5B, written from scratch: INT4 GEMV/GEMM kernels, Flash-Decoding attention, a full 24-layer decode engine and lossless speculative decoding. Verified against Hugging Face in float64 and benchmarked against bandwidth ceilings, PyTorch and llama.cpp, with the quality cost of INT4 measured on HumanEval. v1.0 is the verified baseline; v1.1 is the performance pass.

## Open Source Contributions

**[peft](https://github.com/huggingface/peft)** — Contributions to Hugging Face's PEFT library across its benchmark suite and core code.

- **Image-gen benchmark (FLUX.2-klein):** Added experiments for 9 PEFT methods/variants (BEFT, HiRA, AdaMSS, UniLoRA, VeLoRA, BD-LoRA, MonteCLoRA, TinyLoRA, Super-Tuning), evaluated against LoRA/OFT baselines with learning-rate tuning (Optuna) and ablation sweeps. Implemented ASA training-callback support in the benchmark's `run.py` (previously unimplemented) and extended target-module coverage to single-stream `to_out` layers that the shared config missed. Exploratory and superseded runs are published in the PEFT benchmark graveyard as negative results — including VeLoRA's, where the benchmark's default gradient checkpointing already addresses the activation-memory cost the method targets, a combination the paper never tested.
- **MetaMathQA benchmark:** Added VeLoRA experiment.
- **Bug Fixing:** Fixed incorrect gradients in LoHa with `use_effective_conv2d=True`.

[Discussion #3522](https://github.com/huggingface/peft/discussions/3522) · [Discussion #3521](https://github.com/huggingface/peft/discussions/3521) · [All merged PRs](https://github.com/huggingface/peft/pulls?q=is%3Apr+author%3ASergasgr+state%3Amerged) · [Hugging Face activity](https://huggingface.co/Sergasgr/activity/community)
