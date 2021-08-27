#ifndef UUSSMLMODELS_FORWARDSIMULATION_EIKONET_MODEL_HPP
#define UUSSMLMODELS_FORWARDSIMULATION_EIKONET_MODEL_HPP
#include <memory>
namespace UUSS::ForwardSimulation::EikoNet
{
/// @brief This class is for applying the resnet used to compute travel times.
///        The model is defined in Smith's 2020 paper:
///        EikoNet: Solving the Eikonal equation with Deep Neural Networks.
/// @copyright Ben Baker (University of Utah) distributed under the MIT license.
class Model
{
public:
    Model();

    /// @brief Loads the network weights from an HDF5 file.
    /// @param[in] fileName  The name of the HDF5 file with the model
    ///                      coefficients.
    /// @param[in] verbose   If true then extra information will be printed
    ///                      while loading the weights.  This is useful for
    ///                      debugging.
    /// @throws std::invalid_argument if the HDF5 file does not exist or is
    ///         improperly formatted.
    void loadWeightsFromHDF5(const std::string &fileName,
                             bool verbose = false);
    /// @result True indicates that the model coefficients were set.
    [[nodiscard]] bool haveModelCoefficients() const noexcept;

    /// @brief Computes the travel times from the source to the receiver.
    /// @result The travel time
    template<typename U>
    [[nodiscard]]
    U computeTravelTime(const std::tuple<U, U, U> &sourceLocation,
                        const std::tuple<U, U, U> &receiverLocation) const;

    /// @brief Destructor.
    ~Model();
private:
    class ModelImpl;
    std::unique_ptr<ModelImpl> pImpl;
};
}
#endif
