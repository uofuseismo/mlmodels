#ifndef UUSS_PYTHON_PICKERS_HPP
#define UUSS_PYTHON_PICKERS_HPP
#include <vector>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "uussmlmodels/pickers/cnnOneComponentP/inference.hpp"
#include "uussmlmodels/pickers/cnnThreeComponentS/inference.hpp"
namespace UUSSMLModels::Pickers
{
    namespace CNNOneComponentP
    {   
        class Preprocessing;
    }   
    namespace CNNThreeComponentS
    {   
        class Preprocessing;
    }   
}
namespace UUSSMLModels::Python::Pickers
{
    namespace CNNOneComponentP
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
            std::unique_ptr<UUSSMLModels::Pickers::CNNOneComponentP::Preprocessing> pImpl;
       };
       class Inference
       {
       public:
            Inference();
            void load(const std::string &fileName,
                      UUSSMLModels::Pickers::CNNOneComponentP::Inference::ModelFormat format = UUSSMLModels::Pickers::CNNOneComponentP::Inference::ModelFormat::ONNX);
            [[nodiscard]] bool isInitialized() const noexcept;
            [[nodiscard]] std::pair<double, double> getMinimumAndMaximumPerturbation() const noexcept;
            [[nodiscard]] double getSamplingRate() const noexcept;
            [[nodiscard]] int getExpectedSignalLength() const noexcept;
            [[nodiscard]] double predict(const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &x);
            void clear() noexcept;
            ~Inference();
       private:
            std::unique_ptr<UUSSMLModels::Pickers::CNNOneComponentP::Inference> pImpl;
       };
    }
    void initialize(pybind11::module &m);
}
#endif
