#ifndef UUSS_PYTHON_EVENT_CLASSIFIERS_HPP
#define UUSS_PYTHON_EVENT_CLASSIFIERS_HPP
#include <vector>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
//#include "uussmlmodels/eventClassifiers/cnnThreeComponent/inference.hpp"
namespace UUSSMLModels::EventClassifiers
{
    namespace CNNThreeComponent
    {   
        class Preprocessing;
    }   
}
namespace UUSSMLModels::Python::EventClassifiers
{
    namespace CNNThreeComponent
    {   
        class Preprocessing
        {
        public:
            Preprocessing();
            void clear() noexcept;
            [[nodiscard]] 
            std::tuple<
                pybind11::array_t<double>, //, pybind11::array::c_style | pybind11::array::forcecast>,
                pybind11::array_t<double>, //, pybind11::array::c_style | pybind11::array::forcecast>,
                pybind11::array_t<double> //, pybind11::array::c_style | pybind11::array::forcecast>
            >
            process(const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &vertical, 
                    const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &north,
                    const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &east,
                    double samplingRate = 100);
            pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>
            processVerticalChannel(const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &vertical,
                                   double samplingRate = 100);
            [[nodiscard]] double getScalogramSamplingRate() const noexcept;
            [[nodiscard]] double getScalogramSamplingPeriod() const noexcept;
            [[nodiscard]] pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> getCenterFrequencies() const;
            ~Preprocessing();

            Preprocessing(const Preprocessing &) = delete;
            Preprocessing& operator=(const Preprocessing &) = delete;
       private:
            std::unique_ptr<UUSSMLModels::EventClassifiers::CNNThreeComponent::Preprocessing> pImpl;
       };
    }
    void initialize(pybind11::module &m);
}
#endif
