#ifndef UUSS_ENUMS_HPP
#define UUSS_ENUMS_HPP
namespace UUSS
{
/// @brief Defines the device on which to evaluate the model.
/// @copyright Ben Baker (University of Utah) distributed under the MIT license.
enum class Device
{
    CPU, /*!< Execute the machine learning model on a CPU. */
    GPU  /*!< Execute the machine learning model on a GPU. */
};
}
#endif
