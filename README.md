# Multimodal-RAG-System
## How to run the code

Nvidia-smi version: 12.3\
nvcc -V: 12.3\
pip version: 25.0\
conda version: 23.7.4

#### Redirect to directory
 ```shell
cd Multimodal-RAG-System
```

#### Create env
 ```shell
 chmod +x setup.sh
./setup.sh
conda activate multimodal-rag2
pip install flash-attn --no-build-isolation # Ensure you have already installed ninja using sudo
pip install transformers==4.48.0
pip uninstall -y huggingface-hub
pip install huggingface-hub==0.28.1
pip install --upgrade llama-index-llms-huggingface
 ```

#### Redirect to model folder
 ```shell
 cd model
 ```

#### Start up server on localhost:8000
Ensure test_files only contain PDF files!
 ```shell
 python multimodal_main.py
 ```

#### Create another terminal window to start testing 
> Must redirect to the "model" folder, same directory as "multimodal_main.py"
 ```shell
 python multimodal_test.py
 ```
