#ifndef UUSS_PYTHON_DETECTORS_HPP
#define UUSS_PYTHON_DETECTORS_HPP
#include <vector>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "uussmlmodels/detectors/uNetOneComponentP/inference.hpp"
#include "uussmlmodels/detectors/uNetThreeComponentP/inference.hpp"
#include "uussmlmodels/detectors/uNetThreeComponentS/inference.hpp"
namespace UUSSMLModels::Detectors
{
    namespace UNetThreeComponentP
    {
        class Preprocessing;
    }
    namespace UNetThreeComponentS
    {
        class Preprocessing;
    }
    namespace UNetOneComponentP
    {
        class Preprocessing;
    }
}
namespace UUSSMLModels::Python::Detectors
{
    namespace UNetOneComponentP
    {
        class Preprocessing
        {
        public:
            Preprocessing();
            void clear() noexcept;
            [[nodiscard]] pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>
                 process(const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &x,
                         double samplingRate = 100);
            [[nodiscard]] double getTargetSamplingRate() const noexcept;
            [[nodiscard]] double getTargetSamplingPeriod() const noexcept;
            ~Preprocessing(); 

            Preprocessing(const Preprocessing &) = delete;
            Preprocessing& operator=(const Preprocessing &) = delete;
       private:
            std::unique_ptr<UUSSMLModels::Detectors::UNetOneComponentP::Preprocessing> pImpl;
       };
       class Inference
       {
       public:
            Inference();
            //explicit Inference(const UUSSMLModels::Detectors::UNetOneComponentP::Inference::Device device);
            void load(const std::string &fileName,
                      UUSSMLModels::Detectors::UNetOneComponentP::Inference::ModelFormat format = UUSSMLModels::Detectors::UNetOneComponentP::Inference::ONNX);
            [[nodiscard]] bool isInitialized() const noexcept;
            [[nodiscard]] double getSamplingRate() const noexcept;
            [[nodiscard]] int getMinimumSignalLength() const noexcept;
            [[nodiscard]] int getExpectedSignalLength() const noexcept;
            [[nodiscard]] bool isValidSignalLength(int nSamples) const noexcept;
            [[nodiscard]] std::pair<int, int> getCentralWindowStartEndIndex() const noexcept;
            [[nodiscard]] pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> 
               predictProbability(const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &x,
                                  const bool useSlidingWindow);

            ~Inference();
       private:
            std::unique_ptr<UUSSMLModels::Detectors::UNetOneComponentP::Inference> pImpl;
       };
    }

    namespace UNetThreeComponentP
    {
        class Preprocessing
        {
        public:
            Preprocessing();
            void clear() noexcept;
            [[nodiscard]]
            std::tuple<
                 pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>,
                 pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>,
                 pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> >
                 process(const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &vertical,
                         const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &north,
                         const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &east,
                         double samplingRate = 100);
            [[nodiscard]] double getTargetSamplingRate() const noexcept;
            [[nodiscard]] double getTargetSamplingPeriod() const noexcept;
            ~Preprocessing();

            Preprocessing(const Preprocessing &) = delete;
            Preprocessing& operator=(const Preprocessing &) = delete;
       private:
            std::unique_ptr<UUSSMLModels::Detectors::UNetThreeComponentP::Preprocessing> pImpl;
       };
       class Inference
       {
       public:
            Inference();
            //explicit Inference(const UUSSMLModels::Detectors::UNetThreeComponentP::Inference::Device device);
            void load(const std::string &fileName,
                      UUSSMLModels::Detectors::UNetThreeComponentP::Inference::ModelFormat format = UUSSMLModels::Detectors::UNetThreeComponentP::Inference::ONNX);
            [[nodiscard]] bool isInitialized() const noexcept;
            [[nodiscard]] double getSamplingRate() const noexcept;
            [[nodiscard]] int getMinimumSignalLength() const noexcept;
            [[nodiscard]] int getExpectedSignalLength() const noexcept;
            [[nodiscard]] bool isValidSignalLength(int nSamples) const noexcept;
            [[nodiscard]] std::pair<int, int> getCentralWindowStartEndIndex() const noexcept;
            [[nodiscard]] pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>
               predictProbability(const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &vertical, 
                                  const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &north,
                                  const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &east,
                                  const bool useSlidingWindow);

            ~Inference();
       private:
            std::unique_ptr<UUSSMLModels::Detectors::UNetThreeComponentP::Inference> pImpl;
       };
    }

    namespace UNetThreeComponentS
    {   
        class Preprocessing
        {
        public:
            Preprocessing();
            void clear() noexcept;
            [[nodiscard]]
            std::tuple<
                 pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>,
                 pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>,
                 pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> >
                 process(const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &vertical,
                         const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &north,
                         const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &east,
                         double samplingRate = 100);
            [[nodiscard]] double getTargetSamplingRate() const noexcept;
            [[nodiscard]] double getTargetSamplingPeriod() const noexcept;
            ~Preprocessing();

            Preprocessing(const Preprocessing &) = delete;
            Preprocessing& operator=(const Preprocessing &) = delete;
       private:
            std::unique_ptr<UUSSMLModels::Detectors::UNetThreeComponentS::Preprocessing> pImpl;
       };
       class Inference
       {
       public:
            Inference();
            //explicit Inference(const UUSSMLModels::Detectors::UNetThreeComponentS::Inference::Device device);
            void load(const std::string &fileName,
                      UUSSMLModels::Detectors::UNetThreeComponentS::Inference::ModelFormat format = UUSSMLModels::Detectors::UNetThreeComponentS::Inference::ONNX);
            [[nodiscard]] bool isInitialized() const noexcept;
            [[nodiscard]] double getSamplingRate() const noexcept;
            [[nodiscard]] int getMinimumSignalLength() const noexcept;
            [[nodiscard]] int getExpectedSignalLength() const noexcept;
            [[nodiscard]] bool isValidSignalLength(int nSamples) const noexcept;
            [[nodiscard]] std::pair<int, int> getCentralWindowStartEndIndex() const noexcept;
            [[nodiscard]] pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>
               predictProbability(const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &vertical, 
                                  const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &north,
                                  const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &east,
                                  const bool useSlidingWindow);

            ~Inference();
       private:
            std::unique_ptr<UUSSMLModels::Detectors::UNetThreeComponentS::Inference> pImpl;
       };
    }

    void initialize(pybind11::module &m);
}
#endif
