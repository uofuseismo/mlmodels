#include "uuss/threeComponentPicker/zrunet/model.hpp"
#include <gtest/gtest.h>

namespace
{

TEST(ThreeComponentPicker, ZRUnetCPU)
{
    UUSS::ThreeComponentPicker::ZRUNet::Model<UUSS::Device::CPU> model;
    EXPECT_NO_THROW(model.loadWeightsFromHDF5("data/model_P_006.h5"));
    EXPECT_TRUE(model.haveModelCoefficients());
}

TEST(ThreeComponentPicker, ZRUNetGPU)
{
    try
    {
        UUSS::ThreeComponentPicker::ZRUNet::Model<UUSS::Device::GPU> model;
        EXPECT_NO_THROW(model.loadWeightsFromHDF5("data/model_P_006.h5"));
        EXPECT_TRUE(model.haveModelCoefficients());
    }
    catch (const std::exception &e)
    {
        return;
    }
}

}
