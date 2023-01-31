To convert the 32'nd epoch that was fine-tuned from STEAD to the production model:

    ~/anaconda3/bin/python modelConverter.py -i sDetector_model032.pt -o detectorUNetThreeComponentS.onnx
    git lfs track detectorUNetThreeComponentS.onnx
    git add detectorUNetThreeComponentS.onnx
