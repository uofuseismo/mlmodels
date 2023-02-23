To convert the second epoch that was fine-tuned from California model to the production model:

    ~/anaconda3/bin/python modelConverter.py -i fmPicker_model002.pt -o firstMotionClassifiersCNNOneComponentP.onnx --hdf5_file firstMotionClassifiersCNNOneComponentP.h5
    git lfs track firstMotionClassifiersCNNOneComponentP.onnx
    git add firstMotionClassifiersCNNOneComponentP.onnx
