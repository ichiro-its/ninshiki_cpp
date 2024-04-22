#ifndef NINSHIKI_CPP__CONFIG__GRPC__CALL_DATA_BASE_HPP_
#define NINSHIKI_CPP__CONFIG__GRPC__CALL_DATA_BASE_HPP_

namespace ninshiki_cpp
{
class CallDataBase
{
public:
  CallDataBase();

  virtual void Proceed() = 0;

protected:
  virtual void WaitForRequest() = 0;
  virtual void HandleRequest() = 0;
};
}  // namespace ninshiki_cpp

#endif  // NINSHIKI_CPP__CONFIG__GRPC__CALL_DATA_BASE_HPP_