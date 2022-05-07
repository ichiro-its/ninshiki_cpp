#include "ninshiki_cpp/utils/color.hpp"

namespace ninshiki_cpp::utils
{

Color::Color(
  const std::string & name, int min_hue, int max_hue,
  int min_saturation, int max_saturation, int min_value,
  int max_value) 
: name(name), min_hue(min_hue), max_hue(max_hue),
  min_saturation(min_saturation), max_saturation(max_saturation),
  min_value(min_value), max_value(max_value)
{
}

}  // namespace ninshiki_cpp::utils