#include <fstream>
#include <string>
#include <chrono>
#include <filesystem>
#include <uussmlmodels/firstMotionClassifiers/cnnOneComponentP/inference.hpp>
//#include <uussmlmodels/firstMotionClassifiers/cnnOneComponentP/preprocessing.hpp>
#include <gtest/gtest.h>

namespace
{

using namespace UUSSMLModels::FirstMotionClassifiers::CNNOneComponentP;

/*
std::vector<float> loadTextFile(const std::filesystem::path &textFile)
{
    std::ifstream infile(textFile, std::ios::in);
    std::string line;
    std::vector<float> res;
    res.reserve(360000);
    while (std::getline(infile, line))
    {   
        double t, v;
        sscanf(line.c_str(), "%lf, %lf\n", &t, &v);
        res.push_back(static_cast<float> (v));
    }   
    return res;
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
*/

TEST(FirstMotionClassifiersCNNOneComponentP, Preprocessing)
{

}

}

