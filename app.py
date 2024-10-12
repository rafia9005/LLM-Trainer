import os

os.system("huggingface-cli download TheBloke/zephyr-7B-beta-GGUF zephyr-7b-beta.Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False")
os.system('CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python -q')

from llama_cpp import Llama

llm = Llama(
    model_path="./zephyr-7b-beta.Q4_K_M.gguf",
    n_gpu_layers=-1,  # Uncomment untuk menggunakan akselerasi GPU
    # seed=1337, # Uncomment untuk mengatur seed tertentu
    # n_ctx=2048, # Uncomment untuk memperbesar context window
)

print("Model berhasil dimuat dengan konfigurasi:")
print(f"Model path: {llm.model_path}")
print(f"Context length: {llm.meta_data['llama.context_length']}")
print(f"Embedding length: {llm.meta_data['llama.embedding_length']}")
print(f"Block count: {llm.meta_data['llama.block_count']}")

