//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R A N D O M   U T I L I T I E S
//
//   Artificial Intelligence Techniques, SL
//   artelnics artelnics.com

#pragma once

#include "pch.h"

namespace opennn
{
    void set_seed(Index seed);

    type random_uniform(type min = -1, type max = 1);
    type random_normal(type mean = 0, type std_dev = 1);
    Index random_integer(Index min, Index max);
    bool random_bool(type probability = 0.5);

    void set_random_uniform(Tensor1& tensor, type min = -0.1, type max = 0.1);
    void set_random_uniform(Tensor2& tensor, type min = -0.1, type max = 0.1);

    void set_random_uniform(TensorMap1 tensor, type min = -0.1, type max = 0.1);
    void set_random_uniform(TensorMap2 tensor, type min = -0.1, type max = 0.1);

    void set_random_integer(Tensor2& tensor, Index min, Index max);

    template<typename T>
    void shuffle_vector(vector<T>& vec);

    void shuffle_vector_blocks(vector<Index>& vec, size_t num_parts = 20);

    template<typename T>
    void shuffle_tensor(Tensor<T, 1>& vec);

    Index get_random_element(const vector<Index> &values);
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
