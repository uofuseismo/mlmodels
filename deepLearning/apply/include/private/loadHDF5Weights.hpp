#ifndef PRIVATE_LOADHDF5WEIGHTS_HPP
#define PRIVATE_LOADHDF5WEIGHTS_HPP
#include <string>
#include <hdf5.h>
#include <torch/torch.h>
namespace
{
/*!
 * @brief Helper class to load weights from HDF5.  I'm having linking problems
 *        with HDF5 CXX on CHPC so this aims to alleviate that by wrapping the
 *        C library for this specific applicaiton.
 */
class HDF5Loader
{
public:
    /// Default C'tor
    HDF5Loader() = default;
    /// C'tor that simultaneously opens file
    explicit HDF5Loader(const std::string &fileName)
    {
        openFile(fileName);
    } 
    /// Destructor 
    ~HDF5Loader()
    {
        if (mHaveGroup){H5Gclose(mGroup);}
        if (mHaveFile){H5Fclose(mFile);}
        mHaveGroup = false;
        mHaveFile = false;
    }
    /// Opens an HDF5 file
    void openFile(const std::string &fileName)
    {
        mFile = H5Fopen(fileName.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        mHaveFile = true;
    }
    /// Opens an HDF5 group containing the dataset we're interested in reading
    void openGroup(const std::string &groupName)
    {
        if (!mHaveFile){throw std::runtime_error("File not open");}
        closeGroup();
        auto status = H5Gget_objinfo(mFile, groupName.c_str(), 0, NULL);
        if (status != 0)
        {
            throw std::invalid_argument(groupName + " doesn't exist");
        }
        mGroup = H5Gopen(mFile, groupName.c_str(), H5P_DEFAULT); 
        mHaveGroup = true;
    }
    /// Reads the dataset
    void readDataSet(const std::string &dataSetName,
                     std::vector<hsize_t> *dims, std::vector<float> *values)
    {
        values->clear();
        dims->clear();
        if (!H5Lexists(mGroup, dataSetName.c_str(), H5P_DEFAULT))
        {
            throw std::invalid_argument(dataSetName + " doesn't exist");
        }
        auto dataSet = H5Dopen2(mGroup, dataSetName.c_str(), H5P_DEFAULT);
        auto dataSpace = H5Dget_space(dataSet);
        auto rank = H5Sget_simple_extent_ndims(dataSpace);
        dims->resize(rank);
        H5Sget_simple_extent_dims(dataSpace, dims->data(), NULL);
        hsize_t length = 1;
        for (int i=0; i<static_cast<int> (dims->size()); ++i)
        {
            length = length*dims->at(i);
        }
        // Now read the data
        values->resize(length, 0);
        auto memSpace = H5Screate_simple(rank, dims->data(), NULL);
        auto status = H5Dread(dataSet, H5T_NATIVE_FLOAT, memSpace, dataSpace,
                              H5P_DEFAULT, values->data());
        if (status != 0)
        {
            std::cerr << "Failed to read dataset" << std::endl;
            values->clear();
            dims->clear();
        }
        // Release HDF5 resources
        H5Sclose(memSpace);
        H5Sclose(dataSpace); 
        H5Dclose(dataSet);
    }
    /// Reads the dataset
    void readDataSet(const std::string &dataSetName,
                     std::vector<hsize_t> *dims, std::vector<double> *values)
    {
        values->clear();
        dims->clear();
        if (!H5Lexists(mGroup, dataSetName.c_str(), H5P_DEFAULT))
        {
            throw std::invalid_argument(dataSetName + " doesn't exist");
        }
        auto dataSet = H5Dopen2(mGroup, dataSetName.c_str(), H5P_DEFAULT);
        auto dataSpace = H5Dget_space(dataSet);
        auto rank = H5Sget_simple_extent_ndims(dataSpace);
        dims->resize(rank);
        H5Sget_simple_extent_dims(dataSpace, dims->data(), NULL);
        hsize_t length = 1;
        for (int i=0; i<static_cast<int> (dims->size()); ++i)
        {
            length = length*dims->at(i);
        }
        // Now read the data
        values->resize(length, 0); 
        auto memSpace = H5Screate_simple(rank, dims->data(), NULL);
        auto status = H5Dread(dataSet, H5T_NATIVE_DOUBLE, memSpace, dataSpace,
                              H5P_DEFAULT, values->data());
        if (status != 0)
        {
            std::cerr << "Failed to read dataset" << std::endl;
            values->clear();
            dims->clear();
        }
        // Release HDF5 resources
        H5Sclose(memSpace);
        H5Sclose(dataSpace); 
        H5Dclose(dataSet);
    }
    /// Closes the group
    void closeGroup() noexcept
    {
        if (mHaveGroup){H5Gclose(mGroup);}
        mHaveGroup = false;
    }
    /// Closes the file 
    void closeFile() noexcept
    {
        closeGroup();
        if (mHaveFile){H5Fclose(mFile);}
        mHaveFile = false;
    } 
//private:
    hid_t mFile = 0;
    hid_t mGroup = 0;
    bool mHaveFile = false;
    bool mHaveGroup = false;
};
/// @brief Loads the weights in the convolutional layers.
/// @param[in] loader       The HDF5 data loader with group containing the
///                         dataset already opened.
/// @param[in] dataSetName  The name of the weights to read.
/// @param[in,out] weights  The weights to read.  On input this contains the
///                         expected size.  On exit, this has been filled with 
///                         the coefficients in the HDF5 file in a storage
///                         format appropriate for libtorch and, if necessary,
///                         transferred to the GPU.
/// @param[in] gpu   If true then the weights are to be put onto the GPU.
/// @param[in] verbose  If true then write some information for debugging. 
/// @throws std::invalid_argument if the dataset can't be read or the sizes
///         are inconsistent.
[[maybe_unused]]
void readWeightsFromHDF5(HDF5Loader &loader, 
                         const std::string &dataSetName,
                         torch::Tensor &weights,
                         const bool gpu = false,
                         const bool verbose = false,
                         const int device = -1)
{
    if (verbose){std::cout << "Loading " << dataSetName << std::endl;}
    // Load the data
    std::vector<hsize_t> dims;
    std::vector<float> values; 
    loader.readDataSet(dataSetName, &dims, &values);
    if (values.size() < 1 || dims.size() < 1)
    {
        std::cerr << "Failed to read: " << dataSetName << std::endl;
    }
    // Copy result
    std::vector<int64_t> shape(dims.size());
    for (int i=0; i<static_cast<int> (dims.size()); ++i)
    {
        shape[i] = static_cast<int64_t> (dims[i]);
    }
    torch::Tensor result;    
    result = torch::zeros(shape,
                          torch::TensorOptions()
                          .dtype(torch::kFloat32)
                          .requires_grad(false));
    float *resultPtr = result.data_ptr<float> ();
    std::copy(values.begin(), values.end(), resultPtr);
    // Ease off on some memory usage
    values.clear();
    dims.clear();
    // Verify dimensions match
    auto ndim = weights.dim();
    if (ndim != result.dim())
    {
        throw std::runtime_error("Dimensions do not match\n");
    }
    // Make sure the each dimension's lengths make sense
    auto sizeIn = weights.sizes();
    auto sizeRead = result.sizes();
    bool ltrans = false;
    for (int i=0; i<static_cast<int> (sizeIn.size()); ++i)
    {
        if (sizeIn[i] != sizeRead[i]){ltrans = true;}
    }
    if (verbose && ltrans){std::cout << "Transposing" << std::endl;}
    // TensorFlow and torch are off by a transpose
    if (ltrans && ndim == 3){result.transpose_(0, 2);}
    sizeRead = result.sizes();
    for (int i=0; i<static_cast<int> (sizeIn.size()); ++i)
    {
        if (sizeIn[i] != sizeRead[i])
        {
            auto errmsg = "Size mismatch (Dimension,Expecting,Read) = "
                        + std::to_string(i) + ","
                        + std::to_string(sizeIn[i]) + ","
                        + std::to_string(sizeRead[i]) + ")\n";
            throw std::invalid_argument(errmsg);
        }
    }
    if (gpu)
    {
        if (device >= 0)
        {
            if (verbose)
            {
                std::cout << "Putting weights onto device: " << device
                          << std::endl;
            }
            weights = result.to({torch::kCUDA, device});
        }
        else
        {
            weights = result.to(torch::kCUDA);
        }
    }
    else
    {
        weights = result;
    }
}
/// @brief Convenience function to read weights and bias from HDF5.
/// @param[in] loader       The HDF5 data loader with group containing the
///                         weights and bias already opened.
/// @param[in] dataSetName  The root name of the weights and bias to read.
/// @param[in,out] conv     The 1D convolutional layer to be read. On
///                         input the weight and biases tensor sizes must
///                         be set.  On exit, the weights and biases are
///                         set and, if necessary, offloaded to the device.
/// @param[in] gpu   If true then the weights are to be put onto the GPU.
/// @param[in] verbose  If true then write some information for debugging.
/// @throws std::invalid_argument if the dataset can't be read or the sizes
///         are inconsistent.
[[maybe_unused]]
void readWeightsAndBiasFromHDF5(
    HDF5Loader &loader,
    const std::string &dataSetName,
    torch::nn::Conv1d &conv,
    const bool gpu = false,
    const bool verbose = false,
    const int device = -1)
{
    auto weightName = dataSetName + ".weight";
    auto biasName = dataSetName + ".bias";
    readWeightsFromHDF5(loader, weightName, conv->weight, gpu, verbose, device);
    readWeightsFromHDF5(loader, biasName,   conv->bias, gpu, verbose, device);
}
[[maybe_unused]]
void readWeightsAndBiasFromHDF5(
    HDF5Loader &loader,
    const std::string &dataSetName,
    torch::nn::Linear &fcn,
    const bool gpu = false,
    const bool verbose = false,
    const int device = -1)
{
    auto weightName = dataSetName + ".weight";
    auto biasName = dataSetName + ".bias";
    readWeightsFromHDF5(loader, weightName, fcn->weight, gpu, verbose, device);
    readWeightsFromHDF5(loader, biasName,   fcn->bias, gpu, verbose, device);
}
[[maybe_unused]]
void readWeightsAndBiasFromHDF5(
    HDF5Loader &loader,
    const std::string &dataSetName,
    torch::nn::ConvTranspose1d &uconv,
    const bool gpu = false,
    const bool verbose = false,
    const int device = -1)
{
    auto weightName = dataSetName + ".weight";
    auto biasName = dataSetName + ".bias";
    readWeightsFromHDF5(loader, weightName, uconv->weight, gpu,
                        verbose, device);
    readWeightsFromHDF5(loader, biasName,   uconv->bias, gpu, verbose, device);
}
/// @brief Reads the batch normalization weights from HDF5.
/// @param[in] loader       The HDF5 data loader with group containing the
///                         batch normalization weights already opened.
/// @param[in] dataSetName  The root name of the weights to read.
/// @param[in,out] bn       The 1D batch normalization tensor.  This consists
///                         of weights (gamma), a bias, a running mean, 
///                         and a running variance.
///                         On exit, this has been filled with
///                         the coefficients in the HDF5 file in a storage
///                         format appropriate for libtorch and, if necessary,
///                         transferred to the GPU.
/// @param[in] gpu   If true then the weights are to be put onto the GPU.
/// @param[in] verbose  If true then write some information for debugging.
/// @throws std::invalid_argument if the dataset can't be read or the sizes
///         are inconsistent.
[[maybe_unused]]
void readBatchNormalizationWeightsFromHDF5(
    HDF5Loader &loader,
    const std::string &dataSetName,
    torch::nn::BatchNorm1d &bn,
    const bool gpu = false,
    const bool verbose = false,
    const int device = -1)
{
    auto gammaName = dataSetName + ".weight";
    auto biasName = dataSetName + ".bias";
    auto runningMeanName = dataSetName + ".running_mean";
    auto runningVarianceName = dataSetName + ".running_var";
    readWeightsFromHDF5(loader, gammaName, bn->weight, gpu, verbose, device);
    readWeightsFromHDF5(loader, biasName, bn->bias, gpu, verbose, device);
    readWeightsFromHDF5(loader, runningMeanName,
                        bn->running_mean, gpu, verbose, device);
    readWeightsFromHDF5(loader, runningVarianceName,
                        bn->running_var, gpu, verbose, device);
}

}
#endif
