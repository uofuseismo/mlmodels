#ifndef PUUSSMLMODEL_ONECOMPONENTPICKER_MODEL_HPP
#define PUUSSMLMODEL_ONECOMPONENTPICKER_MODEL_HPP
#include <memory>
#include <pybind11/pybind11.h>
#include <uuss/enums.hpp>
// Forward declarations
namespace UUSS
{
  namespace OneComponentPicker
  {
    namespace ZCNN
    {
      class ProcessData;
      template<UUSS::Device> class Model;
    }
  }
}
namespace PUUSSMLModels::OneComponentPicker
{
namespace ZCNN
{
class Model
{
public:
    Model();
    ~Model()= delete;
    Model& operator=(const Model &model) = delete;
private:
    std::unique_ptr<UUSS::OneComponentPicker::ZCNN::Model<UUSS::Device::CPU>> mModelCPU;
    std::unique_ptr<UUSS::OneComponentPicker::ZCNN::Model<UUSS::Device::GPU>> mModelGPU;
};
void initializeModel(pybind11::module &m);
void initializeProcessing(pybind11::module &m);
}
}
#endif
