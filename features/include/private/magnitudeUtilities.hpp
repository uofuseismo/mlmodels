#ifndef PRIVATE_MAGNITUDEUTILITIES_HPP
#define PRIVATE_MAGNITUDEUTILITIES_HPP
#include <cmath>
#include <algorithm>
namespace
{
template<typename T>
[[nodiscard]] T mean(const int n, const T *__restrict__ x)
{
    if (n == 0){return 0;}
    double xSum = 0;
    for (int i = 0; i < n; ++i)
    {
        xSum = xSum + x[i];
    }
    return static_cast<T> (xSum/n);
}
template<typename T>
[[nodiscard]] T variance(const int n, const T *__restrict__ x)
{
    if (n < 2){return 0;}
    auto xMean = mean(n, x);
    double xSum2 = 0;
    for (int i = 0; i < n; ++i)
    {
        auto resid = x[i] - xMean;
        xSum2 = xSum2 + static_cast<double> (resid*resid);
    }
    return static_cast<T> (xSum2/(n - 1));
}
}
#endif
