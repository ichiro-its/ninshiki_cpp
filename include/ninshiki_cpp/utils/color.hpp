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

#ifndef NINSHIKI_CPP__UTILS__COLOR_HPP_
#define NINSHIKI_CPP__UTILS__COLOR_HPP_

#include <string>

namespace ninshiki_cpp::utils
{

class Color
{
public:
  struct Config
  {
    bool invert_hue = false;
    bool use_lab = false;

    int min_hue = 0;
    int max_hue = 360;
    int min_saturation = 0;
    int max_saturation = 100;
    int min_value = 0;
    int max_value = 100;

    int min_lightness = 0;
    int max_lightness = 255;
    int min_a = -128;
    int max_a = 127;
    int min_b = -128;
    int max_b = 127;
  };

  Color(const std::string & name, const Config & config);

  std::string name;
  Config config;
};

}  // namespace ninshiki_cpp::utils

#endif  // NINSHIKI_CPP__UTILS__COLOR_HPP_
