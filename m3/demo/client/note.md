
# Client
## Run 
```
python client.py batch '<圖片png父級目錄(兩層)>' \
  --recursive \
  --user-message "Perform a diagnostic interpretation of the chest X-ray and precisely delineate the location of any observed pulmonary nodules. Please include a clear initial assessment of whether the CXR appears normal or abnormal." \
  --ground_truth_excel '<某種類型的影像ground_truth execel檔案路徑>' \
  --labels "Normal" "Abnormal" "Unknown" \
  --openai-model "mistral-r1" \
  --openai-endpoint "https://g.ubitus.ai/v1/chat/completions" \
  --openai-api-key "SSdJcVXJcbTweusgA/fws18WX3vBXAnfSr2p6lcKdM7+7C0E4OPsdqJqcRbhKMXJBh+h+wpP0o6U87B5eVHnBg=="
```

single imagle
```
python client.py single '~/working/project/builds/ai/models/monai/cuda/files/small_samples/small_sample_chest_pa/11403050086' \
  --prompt_text "Perform a diagnostic interpretation of the chest X-ray and precisely delineate the location of any observed pulmonary nodules. Please include a clear initial assessment of whether the CXR appears normal or abnormal."
```
