#ifndef UTILITIES_HPP
#define UTILITIES_HPP
#include <iostream>
#include <functional>
#include <numeric>
#include <algorithm>
#include <limits>

namespace
{

template<typename T> T getMaxAbs(const size_t n, const T *__restrict__ v)
{
    auto result = std::minmax_element(v, v + n); 
    auto absoluteMax = std::max( std::abs(*result.first),
                                 std::abs(*result.second) );
    return absoluteMax;
}

/// @brief Min/max rescale each channel and copy.
template<typename T>
void rescaleAndCopy(const size_t n,
                    const T *__restrict__ vertical,
                    float *__restrict__ tensor)
{
    constexpr T eps{10*std::numeric_limits<T>::epsilon()};
    auto zMax = ::getMaxAbs(n, vertical);
    if (zMax > eps)
    {
        auto xnorm = static_cast<T> (1.0/static_cast<double> (zMax));
        std::transform(vertical, vertical + n,
                       tensor,
                       std::bind(std::multiplies<T>(),
                       std::placeholders::_1, xnorm));
    }
    else
    {
        std::fill(tensor, tensor + n, 0); 
    }
}

}
#endif
