To convert the x'th epoch that was fine-tuned from California to the production model:

    ~/anaconda3/bin/python modelConverter.py -i model_007.pt -o pickersCNNOneComponentP.onnx --hdf5_file pickersCNNOneComponentP.h5
    git lfs track pickersCNNOneComponentP.onnx
    git add pickersCNNOneComponentP.onnx
