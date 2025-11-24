// Copyright (c) 2021 ICHIRO ITS
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "ninshiki_cpp/utils/color.hpp"

#include <string>

namespace ninshiki_cpp::utils
{

Color::Color(
  const std::string & name,
  bool invert_hue,
  bool use_lab,
  int min_hue,
  int max_hue,
  int min_saturation,
  int max_saturation,
  int min_value,
  int max_value,
  int min_lightness,
  int max_lightness,
  int min_a,
  int max_a,
  int min_b,
  int max_b)
: name(name),
  invert_hue(invert_hue),
  use_lab(use_lab),
  min_hue(min_hue),
  max_hue(max_hue),
  min_saturation(min_saturation),
  max_saturation(max_saturation),
  min_value(min_value),
  max_value(max_value),
  min_lightness(min_lightness),
  max_lightness(max_lightness),min_a(min_a),
  max_a(max_a),
  min_b(min_b),
  max_b(max_b)
{
}

}  // namespace ninshiki_cpp::utils
