To convert the x'th epoch that was fine-tuned from California to the production model:

    ~/anaconda3/bin/python modelConverter.py -i model_011.pt -o firstMotionClassifiersCNNOneComponentP.onnx --hdf5_file firstMotionClassifiersCNNOneComponentP.h5
    git lfs track firstMotionClassifiersCNNOneComponentP.onnx
    git add firstMotionClassifiersCNNOneComponentP.onnx
