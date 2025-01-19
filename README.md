# Multimodal-RAG-System
## How to run the code

#### Redirect to directory
 ```shell
cd Multimodal-RAG-System
```

#### Create env
 ```shell
 chmod +x setup.sh
./setup.sh
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
