//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   O P E N N N   I M A G E S   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef OPENNN_IMAGES_H
#define OPENNN_IMAGES_H

// System includes

#include <iostream>
#include <fstream>
#include <algorithm>
#include <functional>
#include <limits>
#include <cmath>
#include <ctime>

// Eigen includes

#include "config.h"

using namespace std;
using namespace Eigen;

namespace opennn
{
    // Read 

    void read_bmp_image(const string&, Tensor<type, 3>&);

    // Unsigned char

    void sort_channel(Tensor<unsigned char, 1>&, Tensor<unsigned char, 1>&, const int&);

    // Type

    void reflect_image_x(const ThreadPoolDevice*, const Tensor<type, 3>&, Tensor<type, 3>&);
    void reflect_image_y(const ThreadPoolDevice*, const Tensor<type, 3>&, Tensor<type, 3>&);
    void rotate_image(const ThreadPoolDevice*, const Tensor<type, 3>&, Tensor<type, 3>&, const type&);
    void rescale_image(const ThreadPoolDevice*, Tensor<type, 3>&, TensorMap<Tensor<type, 3>>&, const type&);
    void translate_image(const ThreadPoolDevice*, Tensor<type, 3>&, Tensor<type, 3>&, const Index&);

    //const Eigen::array<bool, 3> reflect_horizontal_dimesions = {false, true, false};
    //const Eigen::array<bool, 3> reflect_vertical_dimesions = {true, false, false};

}

#endif // OPENNN_IMAGES_H