//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B L A N K
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "../opennn/opennn.h"

using namespace opennn;

int main()
{
    try
    {
        cout << "Blank Testing OpenNN" << endl;

        std::cout << "--- SIMD Support Check ---" << std::endl;

#if defined(EIGEN_VECTORIZE_AVX512)
        std::cout << "Target SIMD: AVX-512 (64-byte alignment active)" << std::endl;
#elif defined(EIGEN_VECTORIZE_AVX2)
        std::cout << "Target SIMD: AVX2 (32-byte alignment req, using 64-byte padding)" << std::endl;
#elif defined(EIGEN_VECTORIZE_AVX)
        std::cout << "Target SIMD: AVX" << std::endl;
#elif defined(EIGEN_VECTORIZE_SSE4_2)
        std::cout << "Target SIMD: SSE4.2" << std::endl;
#else
        std::cout << "Target SIMD: None (Scalar Mode - Slow!)" << std::endl;
#endif

#ifdef _OPENMP
        std::cout << "OpenMP: Enabled (Threads: " << omp_get_max_threads() << ")" << std::endl;
#else
        std::cout << "OpenMP: Disabled" << std::endl;
#endif

        cout << "Completed." << endl;

        return 0;
    }
    catch (const exception &e)
    {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
}

// OpenNN: Open Neural Networks Library.
// Copyright (C) Artificial Intelligence Techniques SL.
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
