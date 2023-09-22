#ifndef UUSS_PYTHON_FIRST_MOTION_CLASSIFIERS_HPP
#define UUSS_PYTHON_FIRST_MOTION_CLASSIFIERS_HPP
#include <vector>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "uussmlmodels/firstMotionClassifiers/cnnOneComponentP/inference.hpp"
namespace UUSSMLModels::FirstMotionClassifiers
{
    namespace CNNOneComponentP
    {   
        class Preprocessing;
    }   
}
namespace UUSSMLModels::Python::FirstMotionClassifiers
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
            std::unique_ptr<UUSSMLModels::FirstMotionClassifiers::CNNOneComponentP::Preprocessing> pImpl;
       };
       class Inference
       {
       public:
            Inference();
            void load(const std::string &fileName,
                      UUSSMLModels::FirstMotionClassifiers::CNNOneComponentP::Inference::ModelFormat format = UUSSMLModels::FirstMotionClassifiers::CNNOneComponentP::Inference::ModelFormat::ONNX);
            [[nodiscard]] bool isInitialized() const noexcept;
            [[nodiscard]] double getSamplingRate() const noexcept;
            [[nodiscard]] int getExpectedSignalLength() const noexcept;
            [[nodiscard]] double predict(const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &x);
            void clear() noexcept;
            ~Inference();
       private:
            std::unique_ptr<UUSSMLModels::FirstMotionClassifiers::CNNOneComponentP::Inference> pImpl;
       };
    }
    void initialize(pybind11::module &m);
}
#endif
