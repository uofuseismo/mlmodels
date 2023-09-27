#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
//#include <uussmlmodels/eventClassifiers/cnnThreeComponent/inference.hpp>
#include <uussmlmodels/eventClassifiers/cnnThreeComponent/preprocessing.hpp>
#include <gtest/gtest.h>

namespace
{

using namespace UUSSMLModels::EventClassifiers::CNNThreeComponent;

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> 
loadTextFile(const std::filesystem::path &textFile)
{
    std::ifstream infile(textFile, std::ios::in);
    std::string line;
    std::vector<double> vertical;
    std::vector<double> north;
    std::vector<double> east;
    vertical.reserve(7000);
    north.reserve(7000);
    east.reserve(7000);
    while (std::getline(infile, line))
    {   
        double ti, vi, ni, ei;
        sscanf(line.c_str(), "%lf,%lf,%lf,%lf\n", &ti, &vi, &ni, &ei);
        vertical.push_back(vi);
        north.push_back(ni);
        east.push_back(ei);
    }   
    return std::tuple {vertical, north, east};
}

template<class T>
T infinityNorm(const int n, const T x[], const T y[])
{
    T e8{0};
    for (int i = 0; i < n; ++i)
    {   
        e8 = std::max(std::abs(x[i] - y[i]), e8);
    }   
    return e8; 
}

template<class T>
T infinityNorm(const std::vector<T> &x, const std::vector<T> &y) 
{
    auto n = static_cast<int> (std::min(x.size(), y.size()));
    return infinityNorm(n, x.data(),  y.data());
}

TEST(EventClassifiersCNNThreeComponent, Preprocessing)
{
    constexpr double samplingRate{100};
    std::filesystem::path eqFile{"data/eventClassifiers/cnnThreeComponent/UU.NOQ.HH.ZNE.01_eq.csv"};    
    auto [vertical, north, east] = ::loadTextFile(eqFile);
    EXPECT_EQ(vertical.size(), 3001); 
    Preprocessing processing;
    EXPECT_NEAR(processing.getScalogramSamplingRate(), 25, 1.e-10);
    EXPECT_NEAR(processing.getScalogramSamplingPeriod(), 1./25., 1.e-10);
                 processing.process(vertical, north, east, samplingRate);
}

}
