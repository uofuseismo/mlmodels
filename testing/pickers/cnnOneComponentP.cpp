#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <filesystem>
#include <uussmlmodels/pickers/cnnOneComponentP/inference.hpp>
#include <uussmlmodels/pickers/cnnOneComponentP/preprocessing.hpp>
#include <gtest/gtest.h>

namespace
{

using namespace UUSSMLModels::Pickers::CNNOneComponentP;

std::pair<std::vector<float>, std::vector<float>>
loadWaveformTextFile(const std::filesystem::path &textFile)
{
    std::ifstream infile(textFile, std::ios::in);
    std::string line;
    std::vector<float> times, res;
    times.reserve(400);
    res.reserve(400);
    while (std::getline(infile, line))
    {
        double t, v;
        sscanf(line.c_str(), "%lf, %lf\n", &t, &v);
        times.push_back(static_cast<float> (t));
        res.push_back(static_cast<float> (v));
    }
    return std::pair {times, res};
}

std::vector<std::vector<float>>
loadInputTextFile(const std::filesystem::path &textFile)
{
    std::ifstream infile(textFile, std::ios::in);
    std::string line;
    std::vector<std::vector<float>> signals;
    std::vector<float> signal1, signal2, signal3, signal4;
    signal1.reserve(400);
    signal2.reserve(400);
    signal3.reserve(400);
    signal4.reserve(400);
    while (std::getline(infile, line))
    {
        double v1, v2, v3, v4;
        sscanf(line.c_str(), "%lf, %lf, %lf, %lf\n",
               &v1, &v2, &v3, &v4);
        signal1.push_back(static_cast<float> (v1));
        signal2.push_back(static_cast<float> (v2));
        signal3.push_back(static_cast<float> (v3));
        signal4.push_back(static_cast<float> (v4));
    }
    signals.push_back(std::move(signal1));
    signals.push_back(std::move(signal2));
    signals.push_back(std::move(signal3));
    signals.push_back(std::move(signal4));
    return signals;
}

std::vector<float>
loadOutputTextFile(const std::filesystem::path &textFile)
{
    std::vector<float> solution;
    std::ifstream infile(textFile, std::ios::in);
    std::string line;
    solution.reserve(4);
    while (std::getline(infile, line))
    {
        double v1;
        sscanf(line.c_str(), "%lf\n", &v1);
        solution.push_back(static_cast<float> (v1));
    }
    return solution;
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


TEST(PickersCNNOneComponentP, Preprocessing)
{
    constexpr double samplingRate{100};
    const std::filesystem::path dataDir{"data/pickers/cnnOneComponentP/"};
    // Load input / reference waveforms
    auto [times, vertical] = ::loadWaveformTextFile(dataDir / "uu.gzu.ehz.01.txt");
    EXPECT_EQ(times.size(), 400);
    auto [timesRef, verticalRef]
         = ::loadWaveformTextFile(dataDir / "uu.gzu.ehz.01.proc.txt");
    EXPECT_EQ(times.size(), timesRef.size());

    Preprocessing processing;

    EXPECT_NEAR(processing.getTargetSamplingRate(),   100,  1.e-14);
    EXPECT_NEAR(processing.getTargetSamplingPeriod(), 0.01, 1.e-14);
    auto verticalProc = processing.process(vertical, samplingRate);
    EXPECT_EQ(vertical.size(), verticalProc.size());

    // Tabulate infinity norms
    EXPECT_EQ(verticalRef.size(), verticalProc.size());
    auto e8Vertical = ::infinityNorm(verticalRef, verticalProc);

    EXPECT_NEAR(e8Vertical, 0, 1.e-3);
}

TEST(PickersCNNOneComponentP, InferenceONNX)
{
    // Models
    const std::string onnxFile{"../pickers/cnnOneComponentP/models/pickersCNNOneComponentP.onnx"};
    // Load data and reference solutions
    const std::string dataFile{"data/pickers/cnnOneComponentP/cnnnetTestInputs.txt"};
    const std::string perturbationFile{"data/pickers/cnnOneComponentP/cnnnetTestOutputs.txt"};
    auto signals = ::loadInputTextFile(dataFile);
    auto referencePerturbations = ::loadOutputTextFile(perturbationFile);
    ASSERT_EQ(signals.size(), referencePerturbations.size());

    // Check some basics
    EXPECT_EQ(Inference::getExpectedSignalLength(), 400);
    EXPECT_NEAR(Inference::getSamplingRate(), 100, 1.e-14);
    Inference inference;
    EXPECT_NO_THROW(inference.load(onnxFile, Inference::ModelFormat::ONNX));

    // Validate pick corrections 
    for (int i = 0; i < static_cast<int> (signals.size()); ++i)
    {
        auto perturbation = inference.predict(signals.at(i));
        auto perturbationReference = referencePerturbations.at(i);
        EXPECT_NEAR(std::abs(perturbationReference - perturbation), 0, 1.e-4);
    }
}

TEST(PickersCNNOneComponentP, InferenceHDF5)
{
    // Models
    const std::string h5File{"../pickers/cnnOneComponentP/models/pickersCNNOneComponentP.h5"};
    // Load data and reference solutions
    const std::string dataFile{"data/pickers/cnnOneComponentP/cnnnetTestInputs.txt"};
    const std::string perturbationFile{"data/pickers/cnnOneComponentP/cnnnetTestOutputs.txt"};
    auto signals = loadInputTextFile(dataFile);
    auto referencePerturbations = loadOutputTextFile(perturbationFile);
    ASSERT_EQ(signals.size(), referencePerturbations.size());

    // Load up the H5 model
    Inference inference;
    EXPECT_NO_THROW(inference.load(h5File, Inference::ModelFormat::HDF5));

    // Validate pick corrections 
    for (int i = 0; i < static_cast<int> (signals.size()); ++i)
    {
        auto perturbation = inference.predict(signals.at(i));
        auto perturbationReference = referencePerturbations.at(i);
        EXPECT_NEAR(std::abs(perturbationReference - perturbation), 0, 1.e-4);
    }
}

}

