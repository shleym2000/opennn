//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   O P E N N N   I M A G E S   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include <cstdint>
#include <string>
#include <filesystem>
#include "../eigen/unsupported/Eigen/CXX11/Tensor"

using type = float;

using namespace std;
using namespace Eigen;

namespace opennn
{
uint8_t read_u8(ifstream&, const string&);
uint16_t read_u16_le(ifstream&, const string&);
uint32_t read_u32_le(ifstream&, const string&);
int32_t read_s32_le(ifstream&, const string&);

Tensor3 read_bmp_image(const filesystem::path&);

Tensor3 resize_image(const Tensor3&, const Index&, const Index&);

void reflect_image_x(const ThreadPoolDevice*, Tensor3&);
void reflect_image_y(const ThreadPoolDevice*, Tensor3&);
void rotate_image(const ThreadPoolDevice*, const Tensor3&, Tensor3&, const type&);
void translate_image_x(const ThreadPoolDevice*, const Tensor3&, Tensor3&, const Index&);
void translate_image_y(const ThreadPoolDevice*, const Tensor3&, Tensor3&, const Index&);
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
