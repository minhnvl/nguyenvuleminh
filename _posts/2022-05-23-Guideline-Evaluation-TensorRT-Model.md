## I. Introduction
- The evaluation for models with the different platforms is a big deal for developers. This guideline is able to address this issue. 
- In this guideline, I will show you how to convert the yolov5 model to other platforms and evaluate the models after converting such as TensortRT, ONNX, Torchscript, Openvino.
- With all the enthusiasm, I hope to help everyone get the good job.
:::    warning
***Author: Nguyen Vu Le Minh (MinhNVL-FTECH)***
:::



## II. Convert Yolov5 to Tensorrt

### 1) Download source code

Download yolov5 from github with version v5:
:::    success
- git clone -b v5.0 https://github.com/ultralytics/yolov5.git
:::


Download TensorRT:
:::    success
- git clone -b yolov5-v5.0 https://github.com/wang-xinyu/tensorrtx.git

:::
Download Evaluate model:
:::    success
- git clone https://github.com/rafaelpadilla/review_object_detection_metrics#how-to-use-this-project
:::

### 2) Installation

- Move to folder review_object_detection_metrics:
:::    success
***cd review_object_detection_metrics***
:::


- Create and activate vitural environment by conda:
:::  success
***conda env create -n \<envname> --file environment.yml***
**Ex:** *conda env create -n minhdeptrai --file environment.yml*
    
***conda activate \<envname>***
**Ex:** *conda activate minhdeptrai*
:::

- Move to folder yolov5 to install library:
    
:::    success
***cd .\./yolov5/***
***pip3 install -r requirements.txt***
:::   
    
### 3) Convert Yolov5 model to TensortRT:

- After training, your model will be saved at the folder ***"run/train/exp/weights/"***. 
- To generate the *.wts model, you need to copy file **gen_wts\.py** from folder tensorrtx to folder yolov5. Then, copy the *.wts model to tensorrtx folder. Please, run follow the command line:
    
:::    success
You are in the yolov5 folder.
***cp .\./tensorrtx/yolov5/gen_wts.py ./***
***python3 genwts.py -w run/train/exp/weights/best.pt -o best.wts***
***cd .\./tensorrtx/yolov5/
mkdir build/
cp .\./.\./yolov5/best.wts ./build/***
:::       

- We must modify the class number suitable on our class at **CLASS_NUM** in ***yololayer.h***
     ```
     static constexpr int MAX_OUTPUT_BBOX_COUNT = 1000;
    static constexpr int CLASS_NUM = 3;
    static constexpr int INPUT_H = 640;  // yolov5's input height and width must be divisible by 32.
    static constexpr int INPUT_W = 640;
    ```
    ![](https://i.imgur.com/iySpjd2.png)

- In ***yolov5.cpp*** file, we can change the parameters shown in below fig:
    - If you want to quantize 32b, 16b floating-point or Int8, you need to change **USE_FP32**, **USE_FP16** or **USE_INT8**, respectively
    - Batch size at **BATCH_SIZE** argument.
    - Change threshold parameter if you want.
    - Add more parameter: EXPORT_TXT and WRITE_IMG
   ``` 
    #define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
    #define DEVICE 0  // GPU id
    #define NMS_THRESH 0.4
    #define CONF_THRESH 0.5
    #define BATCH_SIZE 1
    #define MAX_IMAGE_INPUT_SIZE_THRESH 3000 * 3000 // ensure it exceed the maximum size in the input images !
    #define EXPORT_TXT 1 
    #define WRITE_IMG 0 
   ```
    ![](https://i.imgur.com/9ZmttfY.png)
    
    - For **quantization INT8**, you must change training directory shown in the below fig. This will help you get the best result. 
    
    ![](https://i.imgur.com/OQAFuhX.png)

    
    
- In order to evaluation, we need **add more the lines code** to export inference data with the format .txt in ***yolov5.cpp***.
``` 
for (int b = 0; b < fcount; b++) {
            auto& res = batch_res[b];
            nms(res, &prob[b * OUTPUT_SIZE], CONF_THRESH, NMS_THRESH);
        }
        for (int b = 0; b < fcount; b++) {
            auto& res = batch_res[b];
            cv::Mat img = imgs_buffer[b];

            //------ MinhNVL -------
            #if EXPORT_TXT
                std::ofstream txtfile;
                std::system("mkdir -p ./out/");
                std::string img_name =  file_names[f - fcount + 1 + b];
                txtfile.open("./out/" + img_name.substr(0, img_name.find_last_of(".")) + ".txt");
            #endif


            for (size_t j = 0; j < res.size(); j++) {
                cv::Rect r = get_rect(img, res[j].bbox);
                cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);

                #if EXPORT_TXT
                     txtfile << coco_class_mapping[(int) res[j].class_id] - 1  << " " << res[j].conf << " " << r.tl().x << " " << r.tl().y << " " << r.width << " " << r.height << std::endl;
                #endif

            }
            #if EXPORT_TXT
                txtfile.close();
            #endif

            #if WRITE_IMG
                cv::imwrite("_" + file_names[f - fcount + 1 + b], img);
            #endif
        }
```    
![](https://i.imgur.com/jz1HqWc.png)
    
- If this way makes some trouble to you. Don't worry about that. I uploaded the code in the link **xxxxxxx**. You can download and replace on yolov5.cpp file. ***NOTE:*** the code is just suitable with Tensorrtx version 5.0, it can get the error if you use different versions.

    
- We have a lot of ways to convert the yolov5 model to the tensorrtx model. Howerver, in this guideline, I just guide you the common way that is using ***docker image***
    
- We must download the docker image in the url (...) **(Update later)**
- We will run the docker image following:

:::    success
***docker run --gpus "device=1^"^ -it --rm -v $PWD:/yolov5 ducpvftech/21.02-py3-cv-tf-torch-fgradcam:latest /bin/bash***

    
**!!!Note: you can replace $PWD by other path depend on you.**
:::

- Update pip and nvidia-pyindex package
:::    success
***pip3 install --upgrade pip
    pip3 install --upgrade nvidia-tensorrt --no-cache-dir***

:::  
    
- Next, we will go to the build folder and check library and package.
    
:::    success
***cd ./build
cmake ..
make***
:::      

- After checking, we will run the convert best.wts model to best.engine model and detect it.


:::    success


**sudo ./yolov5 -s [.wts] [.engine] [s/m/l/x/n6/s6/m6/l6/x6 or c/c6 gd gw]**  // serialize model to plan file
**sudo ./yolov5 -d [.engine] [image folder]**  // deserialize and run inference, the images in [image folder] will be processed.
     ---
// For example yolov5s
**sudo ./yolov5 -s yolov5s.wts yolov5s.engine s**
**sudo ./yolov5 -d yolov5s.engine ./flag_dataset_v2.2/val/image**
    ---
// For example Custom model with depth_multiple=0.17, width_multiple=0.25 in yolov5.yaml
**sudo ./yolov5 -s yolov5_custom.wts yolov5.engine c 0.17 0.25**
**sudo ./yolov5 -d yolov5.engine ./flag_dataset_v2.2/val/image** 
:::

- After running command line **sudo ./yolov5 -d yolov5.engine ./flag_dataset_v2.2/val/image**. The output will be exported at **out** folder. All the files in **out** folder is .txt file with the format is **\<classid>\<confidence>\<left>\<top>\<width>\<height>**

### 4) Evaluate TensorRT model:

- Return to folder **review_object_detection_metrics** and execute code
:::success
python setup.py install
python run.py
:::
- The interface will be open like this
![](https://i.imgur.com/FtJQAg8.png)

- Following the steps:
    - Create a file .txt classes containt your dataset. **Ex:**
            ![](https://i.imgur.com/Wo9uRIV.png)
            
    - Enter the label directory of your dataset **(1)**
    - Enter the image directory of your dataset **(2)**
    - Enter the classes file (.txt) directory you have just created **(3)**
    - Choose the field YOLO(.txt) **(4)**
    - Enter the output directory which export after detecting by TensorRT **(5)**
    - Enter the classes file (.txt) directory you have just created **(6)**
    - Choose the field **\<classid>\<confidence>\<left>\<top>\<width>\<height>** **(7)**
    - Choose the metrics you want **(8)**
    - Choose your output folder **(9)**
    - Click **Run** button **(10)**

    
######  tags: `FTECH - Network Optimization`
