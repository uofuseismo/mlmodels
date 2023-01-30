To convert the 22'nd epoch that was fine-tuned from STEAD to the production model:

    ~/anaconda3/bin/python modelConverter.py -i oneCompPDetectorMEW_model_022.pt -o detectorsUNetOneComponentP.onnx
    git lfs track detectorsUNetOneComponentP.onnx
    git add detectorsUNetOneComponentP.onnx

