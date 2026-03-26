# History-LM
**Persona-Adaptive Dual-Model Framework for Local Memory Management**
```mermaid
---
config:
  layout: dagre
---
flowchart LR
 subgraph loop["Inference Loop"]
        System["System Prompt with History"]
        History["History"]
        MainLM["Main Inference Model"]
        User["User Prompt"]
        Response["Persona Response"]
        SummLM["Context Summarization Model"]
  end
    System --> MainLM
    BaseSystem["Base System Prompt"] --> System
    History --> System
    User --> MainLM
    MainLM --> Response
    Response --> SummLM
    SummLM --> History

    style MainLM fill:#ff9,stroke:#333,stroke-width:2px,color:#000
    style History fill:#bbf,stroke:#333,stroke-width:2px,color:#000
    style SummLM fill:#ff9,stroke:#333,stroke-width:2px,color:#000
    style loop stroke:#666,stroke-dasharray: 5 5
```

## Features
- **Dual-Model Architecture**: Separates **Main Inference** and **Context Summarization** to maintain long-term memory without VRAM overflow.
- **Soft-Coded Personas**: Easily switch or add AI identities via `SystemPromptDict` without modifying core logic.
- **Memory Efficiency**: Optimized with 4-bit NF4 Quantization to run models on consumer-grade GPUs.
- **Infinite Context**: Automatically condenses dialogue history into a 3-sentence summary for every turn.
- **Hardware Requirements**: Requires CUDA-enabled GPU.
- **Models**: Meta-Llama-3.1-8B & Qwen-0.6B (Default Settings).

## Installation
```bash
pip install torch transformers bitsandbytes accelerate
```
