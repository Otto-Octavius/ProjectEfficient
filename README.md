# Efficient Fine-tuning of LLM on a Single GPU

**Introduction:**

This project centers on optimizing the performance of pretrained Large Language Models (LLMs) by employing efficient fine-tuning techniques. Leveraging Meta AI’s open-source LLaMA2 7B model as a foundational framework, we aim to enhance the adaptability and resource efficiency of LLMs, particularly in environments constrained by computational resources.

## Key Techniques:

**Gradient Checkpointing:**

A strategy involving the periodic saving of model states during training. Checkpointing enables the resumption of training from specific points, thereby mitigating the risk of losing progress due to unexpected interruptions or hardware failures.

**Mixed Precision Training:**

An optimization technique that combines both single-precision and reduced-precision arithmetic during training. By using lower precision for certain computations while maintaining higher precision for critical operations, mixed precision training accelerates training speed and reduces memory requirements.


**Low-Rank Adaptation:**

A method for reducing the computational complexity of pretrained LLMs by approximating their weight matrices with low-rank decompositions. This approach effectively reduces the number of parameters in the model, leading to faster inference and decreased memory usage.
