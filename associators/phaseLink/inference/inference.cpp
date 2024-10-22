#include <iostream>
#include <string>
#include <limits>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <filesystem>
#ifndef NDEBUG
#include <cassert>
#endif
#include "uussmlmodels/associators/phaseLink/inference.hpp"
#include "uussmlmodels/associators/phaseLink/pick.hpp"
#include "uussmlmodels/associators/phaseLink/arrival.hpp"
#include "utahNormalizationHelpers.hpp"
#include "ynpNormalizationHelpers.hpp"
 
#define N_FEATURES 5

using namespace UUSSMLModels::Associators::PhaseLink;
#include "openvino.hpp"

namespace
{

struct PickFeature
{
    double xNormalized;
    double yNormalized;
    double t;
    size_t originalIndex;
    bool isP;
    bool associated{false};
};

int buildPickMatrix(const std::vector<::PickFeature> &picks,
                    const int simulationSize,
                    const int nFeatures,
                    const double timeWindow,
                    std::vector<float> *X)
{
#ifndef NDEBUG
    assert(timeWindow != 0);
#endif
    auto nWork = static_cast<size_t> (simulationSize*nFeatures);
    if (X->size() != nWork){X->resize(nWork);} 
    std::fill(X->begin(), X->end(), 0.0f);
    auto xData = X->data();
    int nPickRows = std::min(static_cast<int> (picks.size()), simulationSize);
    double t0 = 0;
    if (!picks.empty()){t0 = picks[0].t;}
    for (int i = 0; i < nPickRows; ++i)
    {
#ifndef NDEBUG
        assert(static_cast<size_t> (i*nFeatures + 4) < nWork);
#endif
        xData[i*nFeatures + 0] = picks[i].xNormalized;
        xData[i*nFeatures + 1] = picks[i].yNormalized;
        xData[i*nFeatures + 2] = (picks[i].t - t0)/timeWindow;
        xData[i*nFeatures + 3] = picks[i].isP ? 0 : 1;
        xData[i*nFeatures + 4] = 0;
    }
    for (int i = nPickRows; i < simulationSize; ++i)
    {
#ifndef NDEBUG
        assert(static_cast<size_t> (i*nFeatures + 4) < nWork);
#endif
        xData[i*nFeatures + 4] = 1;
    }
/*
for (int i = 0; i < nPickRows + 1; ++i)
{
std::cout << xData[i*nFeatures  ] << "," << xData[i*nFeatures+1] << "," 
          << xData[i*nFeatures+2] << "," << xData[i*nFeatures+3] << "," 
          << xData[i*nFeatures+4] << std::endl;
}
std::cout << nPickRows << std::endl;
*/
    return nPickRows;
}
}

class Inference::InferenceImpl
{
public:
    /// Constructor
    explicit InferenceImpl(const Region region,
                           const Inference::Device device) :
#ifdef WITH_OPENVINO
        mOpenVINO(region, device),
#endif
        mRegion(region),
        mDevice(device)
    {
        if (mRegion == Region::Utah)
        {
            mStationMap = utahStationMap;
            mTimeWindow = UTAH_TIME_NORMALIZATION;
            mSimulationSize = UTAH_SIMULATION_SIZE;
        }
        else
        {
            mStationMap = ynpStationMap;
            mTimeWindow = YELLOWSTONE_TIME_NORMALIZATION;
            mSimulationSize = YELLOWSTONE_SIMULATION_SIZE;
        }
    }
    ::OpenVINOImpl mOpenVINO;
    Region mRegion{Region::Utah};  
    Inference::Device mDevice{Inference::Device::CPU};
    std::map<std::string, std::pair<double, double>> mStationMap;
    double mTimeWindow{120};
    int mSimulationSize{1000};
    bool mUseOpenVINO{false};
    bool mInitialized{false};
};

/// Constructor
Inference::Inference(const Region region) :
    Inference(region, Inference::Device::CPU)
{
}

/// Constructor with given device
Inference::Inference(const Region region, const Inference::Device device) :
    pImpl(std::make_unique<InferenceImpl> (region, device))
{
}

/// Reset class and release memory
void Inference::clear() noexcept
{
    auto region = pImpl->mRegion;
    auto device = pImpl->mDevice;
    pImpl = std::make_unique<InferenceImpl> (region, device);
}

/// Destructor
Inference::~Inference() = default;

void Inference::load(const std::string &fileName,
                     const ModelFormat format)
{
    if (format != ModelFormat::ONNX)
    {
        throw std::runtime_error("Pytorch variant not implemented");
    }
    if (!std::filesystem::exists(fileName))
    {
        throw std::runtime_error(fileName + " does not exist");
    }
#ifdef WITH_OPENVINO
    pImpl->mOpenVINO.load(fileName);
    pImpl->mInitialized = true;
#else
    throw std::runtime_error("Compile with OpenVINO");
#endif
}

bool Inference::isInitialized() const noexcept
{
    return pImpl->mInitialized;
}

template<typename T>
std::vector<T> Inference::predictProbability(const std::vector<T> &X) const
{
    return predictProbability<T> (getSimulationSize(), X);
}

template<typename T>
std::vector<T> Inference::predictProbability(const int nRows,
                                             const std::vector<T> &X) const
{
    if (!isInitialized()){throw std::runtime_error("Not initialized");}
    if (X.empty()){throw std::runtime_error("X is empty");}
    if (nRows > getSimulationSize())
    {
        throw std::invalid_argument("X has too many examples");
    }
    if (static_cast<int> (X.size()) < getNumberOfFeatures())
    {   
        throw std::invalid_argument(
           "Feature matrix must be at least length " 
         + std::to_string(getNumberOfFeatures()));
    }
    if (static_cast<int> (X.size()) > getSimulationSize()*getNumberOfFeatures())
    {
        throw std::invalid_argument("Feature matrix is too large");
    }
    return pImpl->mOpenVINO.predictProbability<T> (nRows, X);
}

template<typename T>
std::vector<int> Inference::predict(const int nRows,
                                    const std::vector<T> &X,
                                    const double thresholdIn) const
{
    if (thresholdIn < 0 || thresholdIn > 1)
    {
        throw std::invalid_argument("Threshold must be in range [0,1]");
    }
    if (nRows < 1)
    {
        throw std::invalid_argument("No examples");
    }
    if (nRows > getSimulationSize())
    {
        throw std::invalid_argument("Too many examples");
    }
    auto probability = predictProbability(nRows, X);
    std::vector<int> result(nRows, 0);
    auto threshold = static_cast<T> (thresholdIn);
    for (int i = 0; i < static_cast<int> (result.size()); ++i)
    {
        result[i] = 0;
        if (probability[i] > threshold){result[i] = 1;}
    }
    return result;
}

template<typename T>
std::vector<int> Inference::predict(const std::vector<T> &X, 
                                    const double thresholdIn) const
{
    return predict<T> (getSimulationSize(), X, thresholdIn);
}

int Inference::getSimulationSize() const noexcept
{
    return pImpl->mSimulationSize;
}

int Inference::getNumberOfFeatures() noexcept
{
    return N_FEATURES;
}

/// Associate
std::vector<std::vector<Arrival>>
Inference::associate(const std::vector<Pick> &picksIn,
                     const int minimumClusterSizeIn,
                     const double threshold) const
{
    if (threshold < 0 || threshold > 1)
    {
        throw std::invalid_argument("Threshold must be in range [0,1]");
    }
    int minimumClusterSize = std::max(1, minimumClusterSizeIn);
    std::vector<std::vector<Arrival>> allArrivals;
    const double timeWindow = pImpl->mTimeWindow;
    std::vector<::PickFeature> picks;
    picks.reserve(picksIn.size());
    for (size_t iPick = 0; iPick < picksIn.size(); ++iPick)
    {
        if (picksIn[iPick].haveNetwork() &&
            picksIn[iPick].haveStation() &&
            picksIn[iPick].haveTime() &&
            picksIn[iPick].havePhase())
        {
            auto name = picksIn[iPick].getNetwork() + "."
                      + picksIn[iPick].getStation();
            auto idx = pImpl->mStationMap.find(name);
            if (idx != pImpl->mStationMap.end())
            {
std::cout << name << " " << idx->first << " " << " " << idx->second.first << " " << idx->second.second << std::endl;
                double xNormalized = idx->second.first;
                double yNormalized = idx->second.second;
                auto phase = picksIn[iPick].getPhase();
                bool isP = true;
                if (phase == Pick::Phase::S){isP = false;}
                ::PickFeature pickFeature{xNormalized,
                                          yNormalized,
                                          picksIn[iPick].getTime(),
                                          iPick,
                                          isP};
                picks.push_back(pickFeature);
            }
            else
            {
 std::cout << "Doesnt exist" << " " << picksIn[iPick].getStation() << std::endl;
            }
        }
    }
    // Done? 
    if (picks.empty()){return allArrivals;}
    // Sort in time order
    std::sort(picks.begin(), picks.end(),
              [](const auto &lhs, const auto &rhs)
             {
                 return lhs.t < rhs.t;
             });
    // Begin association process
    auto nIterations = static_cast<int> (picks.size());
    auto simulationSize = getSimulationSize();
    auto nFeatures = getNumberOfFeatures();
    std::vector<float> X(simulationSize*nFeatures, 0.0f);
for (int k = 0; k < nIterations; ++k)
{
if (nIterations - k < minimumClusterSize){continue;}
std::vector<PickFeature> picksToAssociate(picks.begin() + k, picks.end());
        // Build the pick matrix
        auto nPickRows = ::buildPickMatrix(picksToAssociate, 
                                           simulationSize,
                                           nFeatures,
                                           timeWindow,
                                           &X);
        // Associate
        auto probability = predictProbability<float> (X);
        // Count instances 
        auto clusterSize = std::count_if(probability.begin(),
                                         probability.begin() + nPickRows,
                                         [threshold](const auto p)
                                         {
                                             return p > threshold;
                                         });
if (clusterSize > 0)
{
for (int i = 0; i < probability.size(); ++i)
{
if (probability[i] > threshold)
{
 std::cout << std::setprecision(14) << k << " "<< clusterSize << " " << i << " " << picksIn[picksToAssociate[i].originalIndex].getStation() << " " << picksIn[picksToAssociate[i].originalIndex].getTime() << std::endl;
}
}
}
getchar();
}
getchar();
    for (int k = 0; k < nIterations; ++k)
    {
        // Are we done?
        if (picks.empty()){break;}
        // Gather all picks in time window of this root phase
        std::vector<PickFeature> picksToAssociate;
        const double rootTime = picks.front().t;
        std::copy_if(picks.begin(), picks.end(),
                     std::back_inserter(picksToAssociate),
                     [&](const auto &pick)
                     {
                        return pick.t < rootTime + timeWindow;
                     });
        // This shouldn't happen
        if (picksToAssociate.empty())
        {
#ifndef NDEBUG
            assert(false);
#else
            continue;
#endif
        }
        // Nothing to do - pop the front and begin again
        if (picksToAssociate.size() == 1)
        {
            picks.erase(picks.begin());
            continue;
        }
        // Build the pick matrix
        auto nPickRows = ::buildPickMatrix(picksToAssociate, 
                                           simulationSize,
                                           nFeatures,
                                           timeWindow,
                                           &X);
        // Associate
        auto probability = predictProbability<float> (X);
        // If the first pick isn't related to itself then no way
        if (probability[0] < threshold)
        {
            picks.erase(picks.begin());
            continue;
        }
        //for (int i =0 ; i < nPickRows + 1; ++i){std::cout << probability[i] << std::endl;}
        // Count instances 
        auto clusterSize = std::count_if(probability.begin() + 1,
                                         probability.begin() + nPickRows,
                                         [threshold](const auto p)
                                         {
                                             return p > threshold;
                                         }) + 1;
        if (clusterSize > minimumClusterSize)
        {
            //std::cout << clusterSize << std::endl;
            // Purge the associated picks in this window.  It's actually
            // easier to go backwards since we delete as we go.
            std::vector<Arrival> arrivals;
            int nCopied{0}; 
            for (int i = std::min(static_cast<int> (probability.size()), nPickRows);
                 i >= 0;
                 --i)
            {
                if (probability[i] > threshold)
                {
                    Arrival arrival(picksIn.at(picks.at(i).originalIndex));
                    std::cout << i << " " << picks.at(i).originalIndex << " " << probability[i] << " " << arrival.getStation() << std::endl;
                    arrival.setProbability(probability[i]);
                    arrivals.push_back(arrival);
                    picks.erase(picks.begin() + i);
                    nCopied = nCopied + 1;
                }
            }
#ifndef NDEBUG
            assert(nCopied == clusterSize);
#endif
            // This actually sorts the picks
            std::reverse(arrivals.begin(), arrivals.end());
            // Copy
            allArrivals.push_back(std::move(arrivals));
getchar();
        }
        else
        {
            // Too small - evict first pick and try again
            picks.erase(picks.begin());
        }
    }
    return allArrivals;
}

///--------------------------------------------------------------------------///
///                           Template Instantiation                         ///
///--------------------------------------------------------------------------///
template std::vector<double>
UUSSMLModels::Associators::PhaseLink::Inference::predictProbability(
    const std::vector<double> &) const;
template std::vector<float> 
UUSSMLModels::Associators::PhaseLink::Inference::predictProbability(
    const std::vector<float> &) const;

template std::vector<double>
UUSSMLModels::Associators::PhaseLink::Inference::predictProbability(
    const int, const std::vector<double> &) const;
template std::vector<float>
UUSSMLModels::Associators::PhaseLink::Inference::predictProbability(
    const int, const std::vector<float> &) const;


template std::vector<int>
UUSSMLModels::Associators::PhaseLink::Inference::predict(
    const int, const std::vector<double> &, const double) const;
template std::vector<int>
UUSSMLModels::Associators::PhaseLink::Inference::predict(
    const int, const std::vector<float> &, const double) const;
template std::vector<int>
UUSSMLModels::Associators::PhaseLink::Inference::predict(
    const std::vector<double> &, const double) const;
template std::vector<int>
UUSSMLModels::Associators::PhaseLink::Inference::predict(
    const std::vector<float> &, const double) const;
