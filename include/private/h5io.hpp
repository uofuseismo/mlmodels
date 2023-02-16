#ifndef PRIVATE_H5IO_HPP
#define PRIVATE_H5IO_HPP
#include <vector>
#include <filesystem>
#include <hdf5.h>
namespace
{
/// @class HDF5Loader
/// @brief Utility to load weights from an HDF5 file.
/// @copyright Ben Baker (University of Utah) distributed under the MIT license.
class HDF5Loader
{
public:
    /// Default Constructor.
    HDF5Loader() = default;
    /// Constructor that simultaneously opens a file.
    explicit HDF5Loader(const std::string &fileName)
    {   
        open(fileName);
    }   
    /// Destructor.
    ~HDF5Loader()
    {
        if (mHaveGroup){H5Gclose(mGroup);}
        if (mHaveFile){H5Fclose(mFile);}
        mHaveGroup = false;
        mHaveFile = false;
    }   
    /// Opens an HDF5 file.
    void open(const std::string &fileName)
    {
        if (!std::filesystem::exists(fileName))
        {
            throw std::invalid_argument("HDF5 file " + fileName
                                      + " does not exist");
        }
        mFile = H5Fopen(fileName.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        mHaveFile = true;
    }
    /// Opens an HDF5 group containing the dataset we're interested in reading
    void openGroup(const std::string &groupName)
    {
        if (!mHaveFile){throw std::runtime_error("File not open");}
        closeGroup();
        auto status = H5Gget_objinfo(mFile, groupName.c_str(), 0, nullptr);
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
            throw std::invalid_argument(dataSetName + " does not exist");
        }
        auto dataSet = H5Dopen2(mGroup, dataSetName.c_str(), H5P_DEFAULT);
        auto dataSpace = H5Dget_space(dataSet);
        auto rank = H5Sget_simple_extent_ndims(dataSpace);
        dims->resize(rank);
        H5Sget_simple_extent_dims(dataSpace, dims->data(), nullptr);
        hsize_t length = 1;
        for (int i = 0; i < static_cast<int> (dims->size()); ++i)
        {
            length = length*dims->at(i);
        }
        // Now read the data
        values->resize(length, 0);
        auto memSpace = H5Screate_simple(rank, dims->data(), nullptr);
        auto status = H5Dread(dataSet, H5T_NATIVE_FLOAT, memSpace, dataSpace,
                              H5P_DEFAULT, values->data());
        // Release HDF5 resources
        H5Sclose(memSpace);
        H5Sclose(dataSpace);
        H5Dclose(dataSet);
        if (status != 0)
        {
            values->clear();
            dims->clear();
            throw std::runtime_error("Failed to read dataset: " + dataSetName);
        }
    }
    /// Reads the dataset
    void readDataSet(const std::string &dataSetName,
                     std::vector<hsize_t> *dims,
                     std::vector<double> *values)
    {
        values->clear();
        dims->clear();
        if (!H5Lexists(mGroup, dataSetName.c_str(), H5P_DEFAULT))
        {
            throw std::invalid_argument(dataSetName + " does not exist");
        }
        auto dataSet = H5Dopen2(mGroup, dataSetName.c_str(), H5P_DEFAULT);
        auto dataSpace = H5Dget_space(dataSet);
        auto rank = H5Sget_simple_extent_ndims(dataSpace);
        dims->resize(rank);
        H5Sget_simple_extent_dims(dataSpace, dims->data(), nullptr);
        hsize_t length = 1;
        for (int i = 0; i < static_cast<int> (dims->size()); ++i)
        {
            length = length*dims->at(i);
        }
        // Now read the data
        values->resize(length, 0);
        auto memSpace = H5Screate_simple(rank, dims->data(), nullptr);
        auto status = H5Dread(dataSet, H5T_NATIVE_DOUBLE, memSpace, dataSpace,
                              H5P_DEFAULT, values->data());
        // Release HDF5 resources
        H5Sclose(memSpace);
        H5Sclose(dataSpace);
        H5Dclose(dataSet);
        if (status != 0)
        {
            values->clear();
            dims->clear();
            throw std::runtime_error("Failed to read dataset: " + dataSetName);
        }
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
    hid_t mFile{0};
    hid_t mGroup{0};
    bool mHaveFile{false};
    bool mHaveGroup{false};
};
}
#endif
