#include <fstream>
#include <string>
#include <chrono>
#include <filesystem>
#include "uussmlmodels/firstMotionClassifiers/cnnOneComponentP/inference.hpp"
#include <uussmlmodels/firstMotionClassifiers/cnnOneComponentP/preprocessing.hpp>
#include <gtest/gtest.h>

namespace
{

using namespace UUSSMLModels::FirstMotionClassifiers::CNNOneComponentP;

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
    std::vector<float> signal1, signal2, signal3, signal4, signal5;
    signal1.reserve(400);
    signal2.reserve(400);
    signal3.reserve(400);
    signal4.reserve(400);
    signal5.reserve(400);
    while (std::getline(infile, line))
    {
        double v1, v2, v3, v4, v5;
        sscanf(line.c_str(), "%lf, %lf, %lf, %lf, %lf\n",
               &v1, &v2, &v3, &v4, &v5);
        signal1.push_back(static_cast<float> (v1));
        signal2.push_back(static_cast<float> (v2));
        signal3.push_back(static_cast<float> (v3));
        signal4.push_back(static_cast<float> (v4));
        signal5.push_back(static_cast<float> (v5));
    }
    signals.push_back(std::move(signal1));
    signals.push_back(std::move(signal2));
    signals.push_back(std::move(signal3));
    signals.push_back(std::move(signal4));
    signals.push_back(std::move(signal5));
    return signals;
}

std::vector<std::tuple<float, float, float>>
loadOutputTextFile(const std::filesystem::path &textFile)
{
    std::vector<std::tuple<float, float, float>> solution;
    std::ifstream infile(textFile, std::ios::in);
    std::string line;
    solution.reserve(5);
    while (std::getline(infile, line))
    {
        double v1, v2, v3;
        sscanf(line.c_str(), "%lf, %lf, %lf\n", &v1, &v2, &v3);
        solution.push_back(
            std::tuple {static_cast<float> (v1),
                        static_cast<float> (v2),
                        static_cast<float> (v3)} );
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
    return infinityNorm(n, x.data(),  y.data());
}


TEST(FirstMotionClassifiersCNNOneComponentP, Preprocessing)
{
    constexpr double samplingRate{100};
    const std::filesystem::path dataDir{"data/firstMotionClassifiers/cnnOneComponentP/"};
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
    auto e8Vertical = infinityNorm(verticalRef, verticalProc);

    EXPECT_NEAR(e8Vertical, 0, 1.e-3);
/*
    std::ofstream procFile(dataDir / "uu.gzu.ehz.01.proc.txt");
    for (int i = 0; i < static_cast<int> (times.size()); ++i)
    {
        procFile << times.at(i) << ", " << verticalProc.at(i) << std::endl;
    }
    procFile.close();
*/
}

TEST(FirstMotionClassifiersCNNOneComponentP, InferenceONNX)
{
    // Models
    const std::string onnxFile{"../firstMotionClassifiers/cnnOneComponentP/models/firstMotionClassifiersCNNOneComponentP.onnx"};
    // Load data and reference solutions
    const std::string dataFile{"data/firstMotionClassifiers/cnnOneComponentP/fmnetTestInputs.txt"};
    const std::string probabilityFile{"data/firstMotionClassifiers/cnnOneComponentP/fmnetTestOutputs.txt"};
    auto signals = loadInputTextFile(dataFile);
    auto referenceProbabilities = loadOutputTextFile(probabilityFile);
    ASSERT_EQ(signals.size(), referenceProbabilities.size());
    // Check some basics
    EXPECT_EQ(Inference::getExpectedSignalLength(), 400);
    EXPECT_NEAR(Inference::getSamplingRate(), 100, 1.e-14);
    Inference inference;
    EXPECT_NEAR(inference.getProbabilityThreshold(), 1./3, 1.e-14);
    EXPECT_NO_THROW(inference.load(onnxFile, Inference::ModelFormat::ONNX));

    // Validate probabilities
    for (int i = 0; i < static_cast<int> (signals.size()); ++i)
    {
        auto [pUp, pDown, pUnknown]
            = inference.predictProbability(signals.at(i));
        auto pUpReference      = std::get<0> (referenceProbabilities.at(i));
        auto pDownReference    = std::get<1> (referenceProbabilities.at(i));
        auto pUnknownReference = std::get<2> (referenceProbabilities.at(i));
        EXPECT_NEAR(std::abs(pUpReference - pUp), 0, 1.e-4);
        EXPECT_NEAR(std::abs(pDownReference - pDown), 0, 1.e-4);
        EXPECT_NEAR(std::abs(pUnknownReference - pUnknown), 0, 1.e-4);
    }

    // Now test the classifier
    constexpr double threshold = 0.9;
    inference.setProbabilityThreshold(threshold);
    for (int i = 0; i < static_cast<int> (signals.size()); ++i)
    {
        auto firstMotion = inference.predict(signals.at(i));
        auto pUpReference      = std::get<0> (referenceProbabilities.at(i));
        auto pDownReference    = std::get<1> (referenceProbabilities.at(i));
        auto pUnknownReference = std::get<2> (referenceProbabilities.at(i));
        auto firstMotionReference = Inference::FirstMotion::Unknown;
        if (pUpReference > pUnknownReference &&
            pUpReference > threshold)
        {
            firstMotionReference = Inference::FirstMotion::Up;
        }
        if (pDownReference > pUnknownReference &&
            pDownReference > threshold)
        {
            firstMotionReference = Inference::FirstMotion::Down;
        }
        EXPECT_EQ(firstMotionReference, firstMotion);
    }    
}

TEST(FirstMotionClassifiersCNNOneComponentP, InferenceHDF5)
{
    // Models
    const std::string h5File{"../firstMotionClassifiers/cnnOneComponentP/models/firstMotionClassifiersCNNOneComponentP.h5"};
    // Load data and reference solutions
    const std::string dataFile{"data/firstMotionClassifiers/cnnOneComponentP/fmnetTestInputs.txt"};
    const std::string probabilityFile{"data/firstMotionClassifiers/cnnOneComponentP/fmnetTestOutputs.txt"};
    auto signals = loadInputTextFile(dataFile);
    auto referenceProbabilities = loadOutputTextFile(probabilityFile);
    ASSERT_EQ(signals.size(), referenceProbabilities.size());
    // Check some basics
    EXPECT_EQ(Inference::getExpectedSignalLength(), 400);
    EXPECT_NEAR(Inference::getSamplingRate(), 100, 1.e-14);
    Inference inference;
    EXPECT_NO_THROW(inference.load(h5File, Inference::ModelFormat::HDF5));
    for (int i = 0; i < static_cast<int> (signals.size()); ++i)
    {
        auto [pUp, pDown, pUnknown]
            = inference.predictProbability(signals.at(i));
        auto pUpReference      = std::get<0> (referenceProbabilities.at(i));
        auto pDownReference    = std::get<1> (referenceProbabilities.at(i));
        auto pUnknownReference = std::get<2> (referenceProbabilities.at(i));
        EXPECT_NEAR(std::abs(pUpReference - pUp), 0, 1.e-4);
        EXPECT_NEAR(std::abs(pDownReference - pDown), 0, 1.e-4);
        EXPECT_NEAR(std::abs(pUnknownReference - pUnknown), 0, 1.e-4);
    }
}

}

