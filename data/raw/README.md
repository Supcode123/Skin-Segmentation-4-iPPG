## 📊 Dataset Sources

The data used in this experiment comes from the following sources:

### For Model Training & Inference:
- [The Face Synthetics dataset](https://github.com/microsoft/FaceSynthetics)  
- [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ/tree/master)
  The data is organized in the following structure for training and inference:
  ```
  data_root/
  ├── train             
      ├── images    
      └── labels            
  ├── val             
      ├── images    
      └── labels  
  ├── test           
      ├── images    
      └── labels
  
  ```
  
### For PPGI Application:
- [UBFC-RPPG](https://sites.google.com/view/ybenezeth/ubfcrppg)  
- [PURE](https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/pulse-rate-detection-dataset-pure)  
- KISEMED Institute PPGI Dataset *(internal use only)*

#### UBFC-rPPG:
 ```
   RawData/
   |   |-- subject1/
   |       |-- vid.avi
   |       |-- ground_truth.txt
   |   |-- subject2/
   |       |-- vid.avi
   |       |-- ground_truth.txt
   |...
   |   |-- subjectn/
   |       |-- vid.avi
   |       |-- ground_truth.txt
  
  ```
#### PURE:
 ```
   RawData/
     |   |-- 01-01/
     |      |-- 01-01/
     |      |-- 01-01.json
     |   |-- 01-02/
     |      |-- 01-02/
     |      |-- 01-02.json
     |...
     |   |-- ii-jj/
     |      |-- ii-jj/
     |      |-- ii-jj.json
  
  ```
#### KISEMED Institute PPGI Dataset:
```
   RawData/
     |   |-- p001/
     |      |-- v01/
     |          |-- pictures_ZIP_RAW_RGB
     |          |-- video_RAW_RGB
     |          |-- ...
     |          |-- .scv files
     |      |...
     |      |-- v12/
     |...
     |   |-- p010/
     |      |-- v01/
     |          |-- pictures_ZIP_RAW_RGB
     |          |-- video_RAW_RGB
     |          |-- ...
     |          |-- .scv files
     |      |...
     |      |-- v12/
  
  ```
