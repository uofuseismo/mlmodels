#ifndef PYTHON_BUFFER_HPP
#define PYTHON_BUFFER_HPP
#include <vector>
#include <pybind11/numpy.h>
namespace
{
template<typename T>
[[nodiscard]]
std::vector<T> bufferToVector(const pybind11::buffer_info &buffer)
{
    auto nSamples = static_cast<int> (buffer.size);
    const T *pointer = (T *) (buffer.ptr);
    if (pointer == nullptr)
    {
        throw std::invalid_argument("Buffer data is null");
    }   
    std::vector<T> work(nSamples);
    std::copy(pointer, pointer + nSamples, work.data());
    return work;
}
template<typename T>
[[nodiscard]]
pybind11::array_t<T, pybind11::array::c_style | pybind11::array::forcecast>
   vectorToBuffer(const std::vector<T> &vector)
{
    pybind11::array_t<T, pybind11::array::c_style> buffer(vector.size());
    pybind11::buffer_info bufferHandle = buffer.request();
    auto pointer = static_cast<double *> (bufferHandle.ptr);
    std::copy(vector.begin(), vector.end(), pointer);
    return buffer;
}
}
#endif
