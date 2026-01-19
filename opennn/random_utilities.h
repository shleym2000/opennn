//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R A N D O M   U T I L I T I E S
//
//   Artificial Intelligence Techniques, SL
//   artelnics artelnics.com

#ifndef RANDOMUTILITIES_H
#define RANDOMUTILITIES_H

#include "pch.h"

using namespace std;
using namespace Eigen;

namespace opennn
{
    void set_seed(Index seed);

    type random_uniform(type min = -1, type max = 1);
    type random_normal(type mean = 0, type std_dev = 1);
    Index random_integer(Index min, Index max);
    bool random_bool(type probability = 0.5);

    void set_random_uniform(Tensor1& tensor, type min = -1, type max = 1);
    void set_random_uniform(Tensor2& tensor, type min = -1, type max = 1);

    void set_random_uniform(TensorMap1 tensor, type min = -1, type max = 1);
    void set_random_uniform(TensorMap2 tensor, type min = -1, type max = 1);

    void set_random_integer(Tensor2& tensor, Index min, Index max);

    template<typename T>
    void shuffle_vector(vector<T>& vec);

    template<typename T>
    void shuffle_tensor(Tensor<T, 1>& vec);


    Index get_random_element(const vector<Index> &values);
}

#endif
