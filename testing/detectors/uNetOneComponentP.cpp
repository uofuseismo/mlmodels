#include <fstream>
#include <string>
#include <chrono>
#include <filesystem>
#include <uussmlmodels/detectors/uNetOneComponentP/inference.hpp>
#include <uussmlmodels/detectors/uNetOneComponentP/preprocessing.hpp>
#include <gtest/gtest.h>

namespace
{

using namespace UUSSMLModels::Detectors::UNetOneComponentP;

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

TEST(DetectorsUNetOneComponentP, Preprocessing)
{
    constexpr double samplingRate{100};
    const std::filesystem::path dataDir{"data/detectors/uNetOneComponentP/"};
    // Load input / reference waveforms
    auto vertical = ::loadTextFile(dataDir / "PB.B206.EHZ.zrunet_p.txt");

    auto verticalRef = ::loadTextFile(dataDir / "PB.B206.EHZ.PROC.zrunet_p.txt");

    Preprocessing processing;

    EXPECT_NEAR(processing.getTargetSamplingRate(),   100,  1.e-14);
    EXPECT_NEAR(processing.getTargetSamplingPeriod(), 0.01, 1.e-14);
    auto verticalProc = processing.process(vertical, samplingRate);

    // Tabulate infinity norms
    EXPECT_EQ(verticalRef.size(), verticalProc.size());
    auto e8Vertical = infinityNorm(verticalRef, verticalProc);

    // Some numbers are in the 10,000's so this is fine for floating precision
    EXPECT_NEAR(e8Vertical, 0, 1.e-1);
}

TEST(DetectorsUNetOneComponentP, Inference)
{
    const std::filesystem::path dataDir{"data/detectors/uNetOneComponentP/"};

    auto verticalProc = ::loadTextFile(dataDir / "PB.B206.EHZ.PROC.zrunet_p.txt");

    // Load reference
    auto probaRef = ::loadTextFile(dataDir / "PB.B206.P_PROBA.txt");
    auto probaSlidingRef = ::loadTextFile(dataDir / "PB.B206.P_PROBA.SLIDING.txt");

    const std::string modelName{"../detectors/uNetOneComponentP/models/detectorsUNetOneComponentP.onnx"};
    Inference inference;
    EXPECT_EQ(inference.getExpectedSignalLength(), 1008);
    EXPECT_EQ(inference.getMinimumSignalLength(), 1008);
    EXPECT_NEAR(inference.getSamplingRate(), 100, 1.e-14);
    EXPECT_TRUE(inference.isValidSignalLength(1008));
    EXPECT_FALSE(inference.isValidSignalLength(1009));
    EXPECT_TRUE(inference.isValidSignalLength(1008 + 16*10));

    EXPECT_NO_THROW(inference.load(modelName));
    std::vector<float> proba;
    std::vector<float> verticalProc1008(1008);
    std::copy(verticalProc.begin(), verticalProc.begin() + 1008,
              verticalProc1008.begin());
    EXPECT_NO_THROW(proba = inference.predictProbability(verticalProc1008));
    auto error = infinityNorm(proba.size(), probaRef.data() + 0, proba.data());
    EXPECT_NEAR(error, 0, 5.e-5);
    // Do a big test with a sliding window
    EXPECT_NO_THROW(
        proba = inference.predictProbabilitySlidingWindow(verticalProc));
    EXPECT_EQ(proba.size(), probaSlidingRef.size());
    error = infinityNorm(probaSlidingRef, proba);
    EXPECT_NEAR(error, 0, 5.e-5);
}

}
