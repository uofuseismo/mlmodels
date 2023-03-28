#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <filesystem>
#include <uussmlmodels/pickers/cnnThreeComponentS/inference.hpp>
#include <uussmlmodels/pickers/cnnThreeComponentS/preprocessing.hpp>
#include <gtest/gtest.h>

namespace
{

using namespace UUSSMLModels::Pickers::CNNThreeComponentS;

std::tuple<std::vector<float>,
           std::vector<float>,
           std::vector<float>,
           std::vector<float> >
loadWaveformTextFile(const std::filesystem::path &textFile)
{
    std::ifstream infile(textFile, std::ios::in);
    std::string line;
    std::vector<float> times, vertical, north, east;
    times.reserve(600);
    vertical.reserve(600);
    north.reserve(600);
    east.reserve(600);
    while (std::getline(infile, line))
    {
        double t, z, n, e;
        sscanf(line.c_str(), "%lf, %lf, %lf, %lf\n", &t, &z, &n, &e);
        times.push_back(static_cast<float> (t));
        vertical.push_back(static_cast<float> (z));
        north.push_back(static_cast<float> (n));
        east.push_back(static_cast<float> (e)); 
    }
    return std::tuple {times, vertical, north, east};
}

std::vector<double>
loadOutputTextFile(const std::filesystem::path &textFile)
{
    std::vector<double> solution;
    std::ifstream infile(textFile, std::ios::in);
    std::string line;
    solution.reserve(4);
    while (std::getline(infile, line))
    {   
        double v1; 
        sscanf(line.c_str(), "%lf\n", &v1);
        solution.push_back(v1);
    }   
    return solution;
}

std::vector<
    std::tuple< std::vector<double>,
                std::vector<double>,
                std::vector<double> 
              > >
loadInputTextFile(const std::string &textFile)
{
    std::ifstream infile(textFile, std::ios::in);
    std::string line;
    std::vector<double> z1(600), n1(600), e1(600),
                        z2(600), n2(600), e2(600),
                        z3(600), n3(600), e3(600),
                        z4(600), n4(600), e4(600);
    int i = 0;
    while (std::getline(infile, line))
    {
        double vz1, vn1, ve1;
        double vz2, vn2, ve2;
        double vz3, vn3, ve3;
        double vz4, vn4, ve4;
        sscanf(line.c_str(), "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n",
               &vz1, &vn1, &ve1,
               &vz2, &vn2, &ve2,
               &vz3, &vn3, &ve3,
               &vz4, &vn4, &ve4);
        z1.at(i) = vz1;
        n1.at(i) = vn1;
        e1.at(i) = ve1;
        z2.at(i) = vz2;
        n2.at(i) = vn2;
        e2.at(i) = ve2;
        z3.at(i) = vz3;
        n3.at(i) = vn3;
        e3.at(i) = ve3;
        z4.at(i) = vz4;
        n4.at(i) = vn4;
        e4.at(i) = ve4;
        i = i + 1;
    }
    assert(i == 600);
    std::vector<
        std::tuple< std::vector<double>,
                    std::vector<double>,
                    std::vector<double>
                  > > result;
    std::tuple<std::vector<double>,
               std::vector<double>,
               std::vector<double>> s1{z1, n1, e1};
    std::tuple<std::vector<double>,
               std::vector<double>,
               std::vector<double>> s2{z2, n2, e2};
    std::tuple<std::vector<double>,
               std::vector<double>,
               std::vector<double>> s3{z3, n3, e3};
    std::tuple<std::vector<double>,
               std::vector<double>,
               std::vector<double>> s4{z4, n4, e4};
    result.push_back(std::move(s1));
    result.push_back(std::move(s2));
    result.push_back(std::move(s3));
    result.push_back(std::move(s4));
    return result;
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
    return ::infinityNorm(n, x.data(),  y.data());
}

TEST(PickersCNNThreeComponentS, Preprocessing)
{
    constexpr double samplingRate{100};
    const std::filesystem::path dataDir{"data/pickers/cnnThreeComponentS/"};
    // Load input / reference waveforms
    auto [times, vertical, north, east]
         = ::loadWaveformTextFile(dataDir / "uu.gzu.eh.zne.01.txt");
    EXPECT_EQ(times.size(), 600);

    auto [timesRef, verticalRef, northRef, eastRef]
         = ::loadWaveformTextFile(dataDir / "uu.gzu.eh.zne.01.proc.txt");
    EXPECT_EQ(times.size(), timesRef.size());

    Preprocessing processing;

    EXPECT_NEAR(Inference::getMinimumAndMaximumPerturbation().first,
                -0.85, 1.e-10);
    EXPECT_NEAR(Inference::getMinimumAndMaximumPerturbation().second,
                 0.85, 1.e-10);
    EXPECT_NEAR(processing.getTargetSamplingRate(),   100,  1.e-14);
    EXPECT_NEAR(processing.getTargetSamplingPeriod(), 0.01, 1.e-14);

    auto [verticalProc, northProc, eastProc]
         = processing.process(vertical, north, east, samplingRate);
    EXPECT_EQ(vertical.size(), verticalProc.size());
    EXPECT_EQ(north.size(),    northProc.size());
    EXPECT_EQ(east.size(),     eastProc.size());

    // Tabulate infinity norms
    EXPECT_EQ(verticalRef.size(), verticalProc.size());
    auto e8Vertical = ::infinityNorm(verticalRef, verticalProc);
    auto e8North    = ::infinityNorm(northRef,    northProc);
    auto e8East     = ::infinityNorm(eastRef,     eastProc);

    // There's some pretty big numbers in these lists - like 2000
    EXPECT_NEAR(e8Vertical, 0, 1.e-2);
    EXPECT_NEAR(e8North,    0, 1.e-2);
    EXPECT_NEAR(e8East,     0, 1.e-2);
/*
    std::ofstream procFile(dataDir / "uu.gzu.eh.zne.01.proc.txt");
    for (int i = 0; i < static_cast<int> (times.size()); ++i)
    {
        procFile << times.at(i) << ", "
                 << verticalProc.at(i) << ", "
                 << northProc.at(i) << ", "
                 << eastProc.at(i) << std::endl;
    }
    procFile.close();
*/
}

TEST(PickersCNNThreeComponentS, InferenceONNX)
{
    // Models
    const std::string onnxFile{"../pickers/cnnThreeComponentS/models/pickersCNNThreeComponentS.onnx"};
    // Load data and reference solutions
    const std::string dataFile{"data/pickers/cnnThreeComponentS/cnnnetTestInputs.txt"};
    const std::string perturbationFile{"data/pickers/cnnThreeComponentS/cnnnetTestOutputs.txt"};
    auto signals = ::loadInputTextFile(dataFile);
    auto referencePerturbations = ::loadOutputTextFile(perturbationFile);
    ASSERT_EQ(signals.size(), referencePerturbations.size());

    // Check some basics
    EXPECT_EQ(Inference::getExpectedSignalLength(), 600);
    EXPECT_NEAR(Inference::getSamplingRate(), 100, 1.e-14);
    Inference inference;
    EXPECT_NO_THROW(inference.load(onnxFile, Inference::ModelFormat::ONNX));
    EXPECT_TRUE(inference.isInitialized());

    // Validate pick corrections 
    for (int i = 0; i < static_cast<int> (signals.size()); ++i)
    {
        auto zne = signals.at(i);
        const auto &vertical = std::get<0> (zne);
        const auto &north    = std::get<1> (zne);
        const auto &east     = std::get<2> (zne);
        auto perturbation = inference.predict(vertical, north, east);
        auto perturbationReference = referencePerturbations.at(i);
        EXPECT_NEAR(std::abs(perturbationReference - perturbation), 0, 1.e-4);
    }
}

TEST(PickersCNNThreeComponentS, InferenceHDF5)
{
    // Models
    const std::string h5File{"../pickers/cnnThreeComponentS/models/pickersCNNThreeComponentS.h5"};
    if (!std::filesystem::exists(h5File)){return;}
    // Load data and reference solutions
    const std::string dataFile{"data/pickers/cnnThreeComponentS/cnnnetTestInputs.txt"};
    const std::string perturbationFile{"data/pickers/cnnThreeComponentS/cnnnetTestOutputs.txt"};
    auto signals = ::loadInputTextFile(dataFile);
    auto referencePerturbations = ::loadOutputTextFile(perturbationFile);
    ASSERT_EQ(signals.size(), referencePerturbations.size());

    Inference inference;
    EXPECT_NO_THROW(inference.load(h5File, Inference::ModelFormat::HDF5));
    EXPECT_TRUE(inference.isInitialized());

    // Validate pick corrections 
    for (int i = 0; i < static_cast<int> (signals.size()); ++i)
    {
        auto zne = signals.at(i);
        const auto &vertical = std::get<0> (zne);
        const auto &north    = std::get<1> (zne);
        const auto &east     = std::get<2> (zne);
        auto perturbation = inference.predict(vertical, north, east);
        auto perturbationReference = referencePerturbations.at(i);
        EXPECT_NEAR(std::abs(perturbationReference - perturbation), 0, 1.e-4);
    }   
}

}
