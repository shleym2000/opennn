//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E N S O R   U T I L I T I E S   S O U R C E
//
//   Artificial Intelligence Techniques, SL
//   artelnics@artelnics.com



#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <math.h>
#include <numeric>
#include <stdio.h>



#include "../eigen/Eigen/Dense"

//#include "../eigen/unsupported/Eigen/KroneckerProduct"

#include "strings_utilities.h"
#include "tensors.h"
#include "config.h"

#define GET_VARIABLE_NAME(Variable) (#Variable)

namespace opennn
{


type bound(const type& value, const type& minimum, const type& maximum)
{
    return std::min(std::max(value, minimum), maximum);
}


type calculate_random_uniform(const type& minimum, const type& maximum)
{
    return minimum + (maximum - minimum) * type(rand() / (RAND_MAX + 1.0));
}


bool calculate_random_bool()
{
    return rand() % 2 == 1;
}


template<int rank>
void set_random(Tensor<type, rank>& tensor, const type& minimum, const type& maximum)
{
    random_device rd;
    mt19937 gen(rd());

    uniform_real_distribution<type> distribution(minimum, maximum);

    for (Index i = 0; i < tensor.size(); ++i)
        tensor(i) = distribution(gen);
}


template void set_random(Tensor<type, 1>& tensor, const type& minimum, const type& maximum);
template void set_random(Tensor<type, 2>& tensor, const type& minimum, const type& maximum);
template void set_random(Tensor<type, 3>& tensor, const type& minimum, const type& maximum);
template void set_random(Tensor<type, 4>& tensor, const type& minimum, const type& maximum);


void initialize_sequential(Tensor<type, 1>& vector)
{
    #pragma omp parallel for

    for(Index i = 0; i < vector.size(); i++) 
        vector(i) = type(i);
}


void initialize_sequential(Tensor<Index, 1>& vector)
{
    #pragma omp parallel for

    for(Index i = 0; i < vector.size(); i++) 
        vector(i) = i;
}


void initialize_sequential(Tensor<Index, 1>& new_tensor,
                           const Index& start,
                           const Index& step,
                           const Index& end)
{
    const Index new_size = (end-start)/step+1;

    new_tensor.resize(new_size);
    new_tensor(0) = start;

    for(Index i = 1; i < new_size-1; i++)
        new_tensor(i) = new_tensor(i-1) + step;

    new_tensor(new_size-1) = end;
}


void multiply_rows(Tensor<type, 2>& matrix, const Tensor<type, 1>& vector)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

    #pragma omp parallel for

    for(Index i = 0; i < rows_number; i++)
        for(Index j = 0; j < columns_number; j++)
           matrix(i, j) *= vector(j);
}


void multiply_matrices(const ThreadPoolDevice* thread_pool_device,
                       Tensor<type, 3>& tensor,
                       const Tensor<type, 1>& vector)
{
    const Index rows_number = tensor.dimension(0);
    const Index columns_number = tensor.dimension(1);
    const Index channels = tensor.dimension(2);

    for(Index i = 0; i < channels; i++)
    {
        TensorMap<Tensor<type, 2>> matrix(tensor.data() + i * rows_number * columns_number, rows_number, columns_number);

        matrix.device(*thread_pool_device) = matrix * vector(i);
    }
}


void multiply_matrices(const ThreadPoolDevice* thread_pool_device, Tensor<type, 3>& tensor, const Tensor<type, 2>& matrix)
{
    const Index rows_number = tensor.dimension(0);
    const Index columns_number = tensor.dimension(1);
    const Index channels = tensor.dimension(2);

    for(Index i = 0; i < channels; i++)
    {
        TensorMap<Tensor<type, 2>> slice(tensor.data() + i * rows_number * columns_number, 
                                         rows_number, 
                                         columns_number);

        slice.device(*thread_pool_device) = slice * matrix;
    }
}


void batch_matrix_multiplication(const ThreadPoolDevice* thread_pool_device,
                                 const TensorMap<Tensor<type, 3>>& A,
                                 TensorMap<Tensor<type, 3>>& B,
                                 TensorMap<Tensor<type, 3>>& C,
                                 const Eigen::array<IndexPair<Index>, 1> contraction_axes)
{
    // Assumes A, B & C share dimension 2 and A & B share one of their remaining 2 dimensions (the contraction axes)
    // The other 2 dimensions of C will be the non-equal dimensions of A & B, in that order
    // By default contraction axes are (1, 0)

    const Index A_rows = A.dimension(0);
    const Index A_columns = A.dimension(1);

    const Index B_rows = B.dimension(0);
    const Index B_columns = B.dimension(1);

    const Index C_rows = (contraction_axes[0].first == 0) ? A_columns : A_rows;
    const Index C_columns = (contraction_axes[0].second == 1) ? B_rows : B_columns;

    const Index channels = A.dimension(2);

    type* A_data = (type*)A.data();
    type* B_data = (type*)B.data();
    type* C_data = C.data();

    type* a_matrix_data = nullptr;
    type* b_matrix_data = nullptr;
    type* c_matrix_data = nullptr;

    for(Index i = 0; i < channels; i++)
    {
        a_matrix_data = A_data + A_rows * A_columns * i;
        b_matrix_data = B_data + B_rows * B_columns * i;
        c_matrix_data = C_data + C_rows * C_columns * i;

        const TensorMap<Tensor<type, 2>> A_matrix(a_matrix_data, A_rows, A_columns);
        const TensorMap<Tensor<type, 2>> B_matrix(b_matrix_data, B_rows, B_columns);
        TensorMap<Tensor<type, 2>> C_matrix(c_matrix_data, C_rows, C_columns);

        C_matrix.device(*thread_pool_device) = A_matrix.contract(B_matrix, contraction_axes);
    }
}


void batch_matrix_multiplication(const ThreadPoolDevice* thread_pool_device,
    TensorMap<Tensor<type, 3>>& A,
    const TensorMap<Tensor<type, 3>>& B,
    TensorMap<Tensor<type, 3>>& C,
    const Eigen::array<IndexPair<Index>, 1> contraction_axes)
{
// Assumes A, B & C share dimension 2 and A & B share one of their remaining 2 dimensions (the contraction axes)
// The other 2 dimensions of C will be the non-equal dimensions of A & B, in that order
// By default contraction axes are (1, 0)

    const Index A_rows = A.dimension(0);
    const Index A_columns = A.dimension(1);
    const Index B_rows = B.dimension(0);
    const Index B_columns = B.dimension(1);

    const Index C_rows = (contraction_axes[0].first == 0) ? A_columns : A_rows;
    const Index C_columns = (contraction_axes[0].second == 1) ? B_rows : B_columns;

    const Index channels = A.dimension(2);

    type* A_data = (type*)A.data();
    type* B_data = (type*)B.data();
    type* C_data = C.data();

    type* a_matrix_data = nullptr;
    type* b_matrix_data = nullptr;
    type* c_matrix_data = nullptr;

    for(Index i = 0; i < channels; i++)
    {
        a_matrix_data = A_data + A_rows * A_columns * i;
        b_matrix_data = B_data + B_rows * B_columns * i;
        c_matrix_data = C_data + C_rows * C_columns * i;

        const TensorMap<Tensor<type, 2>> A_matrix(a_matrix_data, A_rows, A_columns);
        const TensorMap<Tensor<type, 2>> B_matrix(b_matrix_data, B_rows, B_columns);
        TensorMap<Tensor<type, 2>> C_matrix(c_matrix_data, C_rows, C_columns);

        C_matrix.device(*thread_pool_device) = A_matrix.contract(B_matrix, contraction_axes);
    }
}


void batch_matrix_multiplication(const ThreadPoolDevice* thread_pool_device,
                                 const Tensor<type, 4>& A,
                                 const Tensor<type, 4>& B,
                                 Tensor<type, 4>& C,
                                 const Eigen::array<IndexPair<Index>, 1> contraction_axes)
{
    // Assumes A, B & C share dimensions 2 & 3 and A & B share one of their remaining 2 dimensions (the contraction axes)
    // The other 2 dimensions of C will be the non-equal dimensions of A & B, in that order
    // By default contraction axes are (1, 0).

    const Index A_rows = A.dimension(0);
    const Index A_columns = A.dimension(1);
    const Index B_rows = B.dimension(0);
    const Index B_columns = B.dimension(1);

    const Index C_rows = (contraction_axes[0].first == 0) ? A_columns : A_rows;
    const Index C_columns = (contraction_axes[0].second == 1) ? B_rows : B_columns;

    const Index channels = A.dimension(2);
    const Index blocks_number = A.dimension(3);

    type* A_data = (type*)A.data();
    type* B_data = (type*)B.data();
    type* C_data = C.data();

    type* a_matrix_data = nullptr;
    type* b_matrix_data = nullptr;
    type* c_matrix_data = nullptr;

    for(Index i = 0; i < blocks_number; i++)
    {
        for(Index j = 0; j < channels; j++)
        {
            a_matrix_data = A_data + A_rows * A_columns * (i * channels + j);
            b_matrix_data = B_data + B_rows * B_columns * (i * channels + j);
            c_matrix_data = C_data + C_rows * C_columns * (i * channels + j);

            const TensorMap<Tensor<type, 2>> A_matrix(a_matrix_data, A_rows, A_columns);
            const TensorMap<Tensor<type, 2>> B_matrix(b_matrix_data, B_rows, B_columns);
            TensorMap<Tensor<type, 2>> C_matrix(c_matrix_data, C_rows, C_columns);

            C_matrix.device(*thread_pool_device) = A_matrix.contract(B_matrix, contraction_axes);
        }
    }
}


void batch_matrix_multiplication(const ThreadPoolDevice* thread_pool_device,
                                 const Tensor<type, 4>& A,
                                 const Tensor<type, 3>& B,
                                 Tensor<type, 4>& C,
                                 const Eigen::array<IndexPair<Index>, 1> contraction_axes)
{
// Assumes A, B & C share their last dimension. A & C also share their second-to-last dimension.
// A & B share one of their remaining 2 dimensions (the contraction axes).
// The other 2 dimensions of C will be the non-equal dimensions of A & B, in that order.
// By default contraction axes are (1, 0).

    const Index A_rows = A.dimension(0);
    const Index A_columns = A.dimension(1);
    const Index B_rows = B.dimension(0);
    const Index B_columns = B.dimension(1);

    const Index C_rows = (contraction_axes[0].first == 0) ? A_columns : A_rows;
    const Index C_columns = (contraction_axes[0].second == 1) ? B_rows : B_columns;

    const Index channels = A.dimension(2);
    const Index blocks_number = A.dimension(3);

    type* A_data = (type*) A.data();
    type* B_data = (type*) B.data();
    type* C_data = C.data();

    type* a_block_data = nullptr;
    type* b_matrix_data = nullptr;
    type* c_block_data = nullptr;

    for(Index i = 0; i < blocks_number; i++)
    {
        a_block_data = A_data + A_rows * A_columns * channels * i;
        b_matrix_data = B_data + B_rows * B_columns * i;
        c_block_data = C_data + C_rows * C_columns * channels * i;

        const TensorMap<Tensor<type, 3>> A_block(a_block_data, A_rows, A_columns, channels);
        const TensorMap<Tensor<type, 2>> B_matrix(b_matrix_data, B_rows, B_columns);
        TensorMap<Tensor<type, 3>> C_block(c_block_data, C_rows, C_columns, channels);

        C_block.device(*thread_pool_device) = A_block.contract(B_matrix, contraction_axes);
    }
}


void batch_matrix_multiplication(const ThreadPoolDevice* thread_pool_device,
                                 const Tensor<type, 4>& A,
                                 const Tensor<type, 3>& B,
                                 Tensor<type, 3>& C,
                                 const Eigen::array<IndexPair<Index>, 1> contraction_axes)
{
    // Assumes A, B & C share their last 2 dimensions, and the first dimension of B is equal to one of the 2 remaining of A (the contraction axes)
    // The other dimension of C will be the non-equal dimension of A
    // By default contraction axes are (1, 0)

    const Index A_rows = A.dimension(0);
    const Index A_columns = A.dimension(1);
    const Index B_rows = B.dimension(0);

    const Index C_rows = (contraction_axes[0].first == 0) ? A_columns : A_rows;

    const Index channels = A.dimension(2);
    const Index blocks_number = A.dimension(3);

    type* A_data = (type*) A.data();
    type* B_data = (type*) B.data();
    type* C_data = C.data();

    type* a_matrix_data = nullptr;
    type* b_vector_data = nullptr;
    type* c_vector_data = nullptr;

    for(Index i = 0; i < blocks_number; i++)
    {
        for(Index j = 0; j < channels; j++)
        {
            a_matrix_data = A_data + A_rows * A_columns * (i * channels + j);
            b_vector_data = B_data + B_rows * (i * channels + j);
            c_vector_data = C_data + C_rows * (i * channels + j);

            const TensorMap<Tensor<type, 2>> A_matrix(a_matrix_data, A_rows, A_columns);
            const TensorMap<Tensor<type, 1>> B_vector(b_vector_data, B_rows);
            TensorMap<Tensor<type, 1>> C_vector(c_vector_data, C_rows);

            C_vector.device(*thread_pool_device) = A_matrix.contract(B_vector, contraction_axes);
        }
    }
}


void batch_matrix_multiplication(const ThreadPoolDevice* thread_pool_device,
                                 const Tensor<type, 4>& A,
                                 const Tensor<type, 3>& B,
                                 TensorMap<Tensor<type, 3>>& C,
                                 const Eigen::array<IndexPair<Index>, 1> contraction_axes)
{
// Assumes A, B & C share their last 2 dimensions, and the first dimension of B is equal to one of the 2 remaining of A (the contraction axes).
// The other dimension of C will be the non-equal dimension of A.
// By default contraction axes are (1, 0).

    const Index A_rows = A.dimension(0);
    const Index A_columns = A.dimension(1);
    const Index B_rows = B.dimension(0);

    const Index C_rows = (contraction_axes[0].first == 0) ? A_columns : A_rows;

    const Index channels = A.dimension(2);
    const Index blocks_number = A.dimension(3);

    type* A_data = (type*)A.data();
    type* B_data = (type*)B.data();
    type* C_data = C.data();

    type* a_matrix_data = nullptr;
    type* b_vector_data = nullptr;
    type* c_vector_data = nullptr;

    for(Index i = 0; i < blocks_number; i++)
    {
        for(Index j = 0; j < channels; j++)
        {
            a_matrix_data = A_data + A_rows * A_columns * (i * channels + j);
            b_vector_data = B_data + B_rows * (i * channels + j);
            c_vector_data = C_data + C_rows * (i * channels + j);

            const TensorMap<Tensor<type, 2>> A_matrix(a_matrix_data, A_rows, A_columns);
            const TensorMap<Tensor<type, 1>> B_vector(b_vector_data, B_rows);
            TensorMap<Tensor<type, 1>> C_vector(c_vector_data, C_rows);

            C_vector.device(*thread_pool_device) = A_matrix.contract(B_vector, contraction_axes);
        }
    }
}


Tensor<type, 2> self_kronecker_product(const ThreadPoolDevice* thread_pool_device, const Tensor<type, 1>& vector)
{
    const Index columns_number = vector.size();

    Tensor<type, 2> matrix(columns_number, columns_number);

    for(Index i = 0; i < columns_number; i++)
    {
        TensorMap<Tensor<type, 1>> column = tensor_map(matrix, i);

        column.device(*thread_pool_device) = vector * vector(i);
    }

    return matrix;
}


void divide_columns(const ThreadPoolDevice* thread_pool_device, Tensor<type, 2>& matrix, const Tensor<type, 1>& vector)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

    type* matrix_data = matrix.data();

    for(Index j = 0; j < columns_number; j++)
    {
        TensorMap<Tensor<type, 1>> column(matrix_data + j*rows_number, rows_number);

        column.device(*thread_pool_device) = column / vector;
    }
}


void divide_columns(const ThreadPoolDevice* thread_pool_device, TensorMap<Tensor<type, 2>>& matrix, const Tensor<type, 1>& vector)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

    type* matrix_data = matrix.data();

    for(Index j = 0; j < columns_number; j++)
    {
        TensorMap<Tensor<type, 1>> column(matrix_data + j*rows_number, rows_number);

        column.device(*thread_pool_device) = column / vector;
    }
}


void sum_columns(const ThreadPoolDevice* thread_pool_device, const Tensor<type, 1>& vector, Tensor<type, 2>& matrix)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

    for(Index i = 0; i < columns_number; i++)
    {
        TensorMap<Tensor<type, 1>> column(matrix.data() + i*rows_number, rows_number);

        column.device(*thread_pool_device) = column + vector(i);
    }
}


void sum_columns(const ThreadPoolDevice* thread_pool_device, const Tensor<type, 1>& vector, TensorMap<Tensor<type, 2>>& matrix)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

    type* matrix_data = matrix.data();

    for(Index i = 0; i < columns_number; i++)
    {
        TensorMap<Tensor<type, 1>> column(matrix_data + i * rows_number, rows_number);

        column.device(*thread_pool_device) = column + vector(i);
    }
}


void sum_matrices(const ThreadPoolDevice* thread_pool_device, const Tensor<type, 1>& vector, Tensor<type, 3>& tensor)
{
    const Index rows_number = tensor.dimension(0);
    const Index columns_number = tensor.dimension(1);
    const Index channels = tensor.dimension(2);

    type* tensor_data = tensor.data();

    const Index slice_size = rows_number * columns_number;

    for(Index i = 0; i < channels; i++)
    {
        TensorMap<Tensor<type,2>> matrix(tensor_data + i*slice_size, rows_number, columns_number);

        matrix.device(*thread_pool_device) = matrix + vector(i);
    }
}


void sum_matrices(const ThreadPoolDevice* thread_pool_device, const TensorMap<Tensor<type, 1>>& vector, Tensor<type, 3>& tensor)
{
    const Index rows_number = tensor.dimension(0);
    const Index columns_number = tensor.dimension(1);
    const Index channels = tensor.dimension(2);

    for(Index i = 0; i < channels; i++)
    {
        TensorMap<Tensor<type,2>> matrix(tensor.data() + i*rows_number*columns_number, rows_number, columns_number);

        matrix.device(*thread_pool_device) = matrix + vector(i);
    }
}


void sum_matrices(const ThreadPoolDevice* thread_pool_device, const Tensor<type, 2>& matrix, Tensor<type, 3>& tensor)
{
    const Index rows_number = tensor.dimension(0);
    const Index columns_number = tensor.dimension(1);
    const Index channels = tensor.dimension(2);

    for(Index i = 0; i < channels; i++)
    {
        TensorMap<Tensor<type,2>> submatrix(tensor.data() + i*rows_number*columns_number, rows_number, columns_number);

        submatrix.device(*thread_pool_device) += matrix;
    }
}


void substract_columns(const ThreadPoolDevice* thread_pool_device, const Tensor<type, 1>& vector, Tensor<type, 2>& matrix)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

    type* matrix_data = matrix.data();

    for(Index i = 0; i < columns_number; i++)
    {
        TensorMap<Tensor<type, 1>> column(matrix_data + i*rows_number, rows_number);

        column.device(*thread_pool_device) = column - vector;
    }
}


void substract_matrices(const ThreadPoolDevice* thread_pool_device, const Tensor<type, 2>& matrix, Tensor<type, 3>& tensor)
{
    const Index rows_number = tensor.dimension(0);
    const Index columns_number = tensor.dimension(1);
    const Index channels = tensor.dimension(2);

    for(Index i = 0; i < channels; i++)
    {
        TensorMap<Tensor<type, 2>> slice(tensor.data() + i * rows_number * columns_number, 
                                         rows_number, 
                                         columns_number);

        slice.device(*thread_pool_device) = slice - matrix;
    }
}


bool is_zero(const Tensor<type, 1>& tensor, const type& limit)
{
    const Index size = tensor.size();

    for(Index i = 0; i < size; i++)
        if(abs(tensor[i]) > type(limit)) 
            return false;

    return true;
}


bool is_false(const Tensor<bool, 1>& tensor)
{
    const Index size = tensor.size();

    for(Index i = 0; i < size; i++)
        if(tensor(i)) 
            return false;

    return true;
}


Index count_true(const Tensor<bool, 1>& tensor)
{
    Index count = 0;

    #pragma omp parallel for reduction(+: count)

    for(int i = 0; i < tensor.size(); i++)
        if(tensor(i))
            count++;

    return count;
}


bool is_binary_vector(const Tensor<type, 1>& vector)
{
    const Index size = vector.size();

    for(Index i = 0; i < size; i++)
        if(vector(i) != type(0) && vector(i) != type(1) && !isnan(vector(i))) 
            return false;

    return true;
}


bool is_binary_matrix(const Tensor<type, 2>& matrix)
{
    const Index size = matrix.size();

    for(Index i = 0; i < size; i++)
        if(matrix(i) != type(0) && matrix(i) != type(1) && !isnan(matrix(i))) 
            return false;

    return true;
}


bool is_constant_vector(const Tensor<type, 1>& vector)
{
    const Index size = vector.size();

    Index first_non_nan_index = 0;

    while (first_non_nan_index < size && isnan(vector[first_non_nan_index])) 
        first_non_nan_index++;
   
    if (first_non_nan_index == size) 
        return true;

    type first_not_nan_element = vector[first_non_nan_index];

    for (Index i = first_non_nan_index + 1; i < size; ++i)
        if (!isnan(vector[i]) && abs(first_not_nan_element - vector[i]) > std::numeric_limits<float>::min())
            return false;

    return true;
}


bool is_constant_matrix(const Tensor<type, 2>& matrix)
{// @todo copy old code
/*
    const Index size = matrix.size();

    Index first_non_nan_index = 0;

    while (first_non_nan_index < size && isnan(matrix[first_non_nan_index])) 
        first_non_nan_index++;
    
    if (first_non_nan_index == size) 
        return true;

    type first_not_nan_element = matrix[first_non_nan_index];

    for (Index i = first_non_nan_index + 1; i < size; ++i)
        if (!isnan(matrix[i]) && abs(first_not_nan_element - matrix[i]) > std::numeric_limits<float>::min())
            return false;
*/
    return true;
}


bool is_equal(const Tensor<type, 2>& matrix, const type& value, const type& tolerance)
{
    const Index size = matrix.size();

    for(Index i = 0; i < size; i++)
        if(abs(matrix(i) - value) > tolerance) 
            return false;

    return true;
}


bool are_equal(const Tensor<type, 1>& vector_1, const Tensor<type, 1>& vector_2, const type& tolerance)
{
    const Index size = vector_1.size();

    for(Index i = 0; i < size; i++)
        if(abs(vector_1(i) - vector_2(i)) > tolerance) 
            return false;

    return true;
}


bool are_equal(const Tensor<bool, 1>& vector_1, const Tensor<bool, 1>& vector_2)
{
    const Index size = vector_1.size();

    for(Index i = 0; i < size; i++)
        if(vector_1(i) != vector_2(i)) 
            return false;

    return true;
}


bool are_equal(const Tensor<type, 2>& matrix_1, const Tensor<type, 2>& matrix_2, const type& tolerance)
{
    const Index size = matrix_1.size();

    for(Index i = 0; i < size; i++)
        if(abs(matrix_1(i) - matrix_2(i)) > tolerance) 
            return false;

    return true;
}


bool are_equal(const Tensor<bool, 2>& matrix_1, const Tensor<bool, 2>& matrix_2)
{
    const Index size = matrix_1.size();

    for(Index i = 0; i < size; i++)
        if(matrix_1(i) != matrix_2(i)) 
            return false;

    return true;
}


Tensor<bool, 2> elements_are_equal(const Tensor<type, 2>& x, const Tensor<type, 2>& y)
{
    if(x.size() != y.size() || x.dimension(0) != y.dimension(0) || x.dimension(1) != y.dimension(1))
        throw runtime_error("Input vectors must have equal sizes.\n");

    Tensor<bool, 2> result(x.dimension(0), x.dimension(1));

    #pragma omp parallel for

    for(int i = 0; i < x.size(); i++) 
        result(i) = (x(i) == y(i)); 

    return result;
}


void save_csv(const Tensor<type,2>& data, const string& filename)
{
    ofstream file(filename);

    if(!file.is_open())
      throw runtime_error("Cannot open matrix data file: " + filename + "\n");

    file.precision(20);

    const Index data_rows = data.dimension(0);
    const Index data_columns = data.dimension(1);

    char separator_string = ';';

    for(Index i = 0; i < data_rows; i++)
    {
       for(Index j = 0; j < data_columns; j++)
       {
           file << data(i, j);

           if(j != data_columns -1)
               file << separator_string;
       }

       file << endl;
    }

    file.close();
}


Tensor<Index, 1> calculate_rank_greater(const Tensor<type, 1>& vector)
{
    const Index size = vector.size();

    Tensor<Index, 1> rank(size);
    iota(rank.data(), rank.data() + rank.size(), 0);

    sort(rank.data(),
         rank.data() + rank.size(),
         [&](Index i, Index j){return vector[i] > vector[j];});

    return rank;
}


Tensor<Index, 1> calculate_rank_less(const Tensor<type, 1>& vector)
{
    const Index size = vector.size();

    Tensor<Index, 1> rank(size);
    iota(rank.data(), rank.data() + rank.size(), 0);

    sort(rank.data(),
         rank.data() + rank.size(),
         [&](Index i, Index j){return vector[i] < vector[j];});

    return rank;
}


void scrub_missing_values(Tensor<type, 2>& matrix, const type& value)
{
    replace_if(matrix.data(), matrix.data()+matrix.size(), [](type x){return isnan(x);}, value);
}


Tensor<string, 1> sort_by_rank(const Tensor<string,1>&tokens, const Tensor<Index,1>&rank)
{
    const Index tokens_size = tokens.size();

    if(tokens_size != rank.size())
        throw runtime_error("Tokens and rank size must be the same.\n");

    Tensor<string,1> sorted_tokens(tokens_size);

    #pragma omp parallel for

    for(Index i = 0; i < tokens_size; i++)
        sorted_tokens(i) = tokens(rank(i));

    return sorted_tokens;
}


Tensor<Index, 1> sort_by_rank(const Tensor<Index,1>&tokens, const Tensor<Index,1>&rank)
{
    const Index tokens_size = tokens.size();

    if(tokens_size != rank.size())
        throw runtime_error("Tokens and rank size must be the same.\n");

    Tensor<Index,1> sorted_tokens(tokens_size);

    #pragma omp parallel for

    for(Index i = 0; i < tokens_size; i++)
        sorted_tokens(i) = tokens(rank(i));

    return sorted_tokens;
}


Index count_less_than(const Tensor<Index,1>& vector, const Index& bound)
{
    Index count = 0;

    #pragma omp parallel for reduction(+: count)

    for(Index i = 0; i < vector.size(); i++)
        if(vector(i) < bound)
            count++;

    return count;
}


Tensor<Index, 1> get_indices_less_than(const Tensor<Index,1>& vector, const Index& bound)
{
    const Index indices_size = count_less_than(vector, bound);

    Tensor<Index, 1> indices(indices_size);

    Index index = 0;

    for(Index i  = type(0); i < vector.size(); i++)
        if(vector(i) < bound)
            indices(index++) = i;

    return indices;
}



Index count_less_than(const Tensor<double,1>& vector, const double& bound)
{
    Index count = 0;

    #pragma omp parallel for reduction(+: count)

    for(Index i = 0; i < vector.size(); i++)
        if(vector(i) < bound)
            count++;

    return count;
}


Tensor<Index, 1> get_indices_less_than(const Tensor<double,1>& vector, const double& bound)
{
    const Index indices_size = count_less_than(vector, bound);

    Tensor<Index, 1> indices(indices_size);

    Index index = 0;

    for(Index i  = type(0); i < vector.size(); i++)
         if(vector(i) < bound)
             indices(index++) = i;

    return indices;
}


Index count_greater_than(const Tensor<Index,1>& vector, const Index& bound)
{
    Index count = 0;

    #pragma omp parallel for reduction(+: count)

    for(Index i = 0; i < vector.size(); i++)
        if(vector(i) > bound)
            count++;

    return count;
}


Tensor<Index, 1> get_elements_greater_than(const Tensor<Index,1>& vector, const Index& bound)
{
    const Index indices_size = count_greater_than(vector, bound);

    Tensor<Index, 1> indices(indices_size);

    Index index = 0;

    for(Index i  = type(0); i < vector.size(); i++)
         if(vector(i) > bound)
             indices(index++) = vector(i);

    return indices;
}


Tensor<Index, 1> get_elements_greater_than(const Tensor<Tensor<Index, 1>,1>& vectors, const Index& bound)
{
    const Index vectors_number = vectors.size();

    Tensor<Index, 1> indices(0);

    for(Index i = 0; i < vectors_number; i++)
    {
        const Tensor<Index, 1> indices_vector = get_elements_greater_than(vectors(i), bound);

        indices = join_vector_vector(indices, indices_vector);
    }

    return indices;
}


void delete_indices(Tensor<string,1>& vector, const Tensor<Index,1>& indices)
{
    const Index original_size = vector.size();

    const Index new_size = vector.size() - indices.size();

    Tensor<string,1> vector_copy(vector);

    vector.resize(new_size);

    Index index = 0;

    for(Index i = 0; i < original_size; i++)
        if(!contains(indices, i))
            vector(index++) = vector_copy(i);
}


void delete_indices(Tensor<Index,1>& vector, const Tensor<Index,1>& indices)
{
    const Index original_size = vector.size();

    const Index new_size = vector.size() - indices.size();

    Tensor<Index,1> vector_copy(vector);

    vector.resize(new_size);

    Index index = 0;

    for(Index i = 0; i < original_size; i++)
        if(!contains(indices, i))
            vector(index++) = vector_copy(i);
}


void delete_indices(Tensor<double,1>& vector, const Tensor<Index,1>& indices)
{
    const Index original_size = vector.size();

    const Index new_size = vector.size() - indices.size();

    Tensor<double,1> vector_copy(vector);

    vector.resize(new_size);

    Index index = 0;

    for(Index i = 0; i < original_size; i++)
        if(!contains(indices, i))
            vector(index++) = vector_copy(i);
}


//Tensor<string, 1> get_first(const Tensor<string,1>& vector, const Index& index)
//{
//    Tensor<string, 1> new_vector(index);

//    copy(vector.data(), vector.data() + index, new_vector.data());

//    return new_vector;
//}


//Tensor<Index, 1> get_first(const Tensor<Index,1>& vector, const Index& index)
//{
//    Tensor<Index, 1> new_vector(index);

//    copy(vector.data(), vector.data() + index, new_vector.data());

//    return new_vector;
//}


Index count_between(const Tensor<type, 1>& vector,const type& minimum, const type& maximum)
{
    const Index size = vector.size();

    Index count = 0;

    #pragma omp parallel for reduction(+: count)

    for(Index i = 0; i < size; i++)
        if(vector(i) >= minimum && vector(i) <= maximum) 
            count++;

    return count;
}


void get_row(Tensor<type, 1>& row, const Tensor<type, 2, RowMajor>& matrix, const Index& row_index)
{
    const Index columns_number = row.dimension(0);

    memcpy(row.data(), matrix.data() + row_index * columns_number, columns_number*sizeof(type));
}


void set_row(Tensor<type,2>& matrix, const Tensor<type, 1>& new_row, const Index& row_index)
{
    const Index columns_number = new_row.size();

    #pragma omp parallel for    

    for(Index i = 0; i < columns_number; i++)
        matrix(row_index, i) = new_row(i);
}


void set_row(Tensor<type, 2, RowMajor>& matrix, const Tensor<type, 1>& vector, const Index& row_index)
{
    const Index columns_number = vector.size();

    memcpy(matrix.data() + row_index * columns_number, vector.data(), columns_number*sizeof(type));
}


Tensor<type,2> filter_column_minimum_maximum(Tensor<type,2>& matrix, 
                                             const Index& column_index, 
                                             const type& minimum, 
                                             const type& maximum)
{
    const Tensor<type, 1> column = matrix.chip(column_index,1);
    const Index new_rows_number = count_between(column, minimum, maximum);

    if(new_rows_number == 0) return Tensor<type,2>();

    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

    bool check_conditions = false;

    Tensor<type,2> new_matrix(new_rows_number, columns_number);

    Index row_index = 0;

    for(Index i = 0; i < rows_number; i++)
    {
        if(matrix(i, column_index) >= minimum 
        && matrix(i, column_index) <= maximum)
        {
            const Tensor<type, 1> row = matrix.chip(i, 0);

            set_row(new_matrix, row, row_index);

            row_index++;

            check_conditions = true;
        }
    }

    if(!check_conditions)
        throw runtime_error("Invalid conditions\n");

    return new_matrix;
}


type l1_norm(const ThreadPoolDevice* thread_pool_device, const Tensor<type, 1>& vector)
{
    Tensor<type, 0> norm;

    norm.device(*thread_pool_device) = vector.abs().sum();

    return norm(0);
}


void l1_norm_gradient(const ThreadPoolDevice* thread_pool_device, const Tensor<type, 1>& vector, Tensor<type, 1>& gradient)
{
    gradient.device(*thread_pool_device) = vector.sign();
}


void l1_norm_hessian(const ThreadPoolDevice*, const Tensor<type, 1>&, Tensor<type, 2>& hessian)
{
    hessian.setZero();
}


type l2_norm(const ThreadPoolDevice* thread_pool_device, const Tensor<type, 1>& vector)
{
    Tensor<type, 0> norm;

    norm.device(*thread_pool_device) = vector.square().sum().sqrt();

    if(isnan(norm(0)))
        throw runtime_error("OpenNN Exception: l2 norm of vector is not a number.\n");

    return norm(0);
}


void l2_norm_gradient(const ThreadPoolDevice* thread_pool_device, const Tensor<type, 1>& vector, Tensor<type, 1>& gradient)
{
    const type norm = l2_norm(thread_pool_device, vector);

    if(norm < type(NUMERIC_LIMITS_MIN))
    {
        gradient.setZero();

        return;
    }

    gradient.device(*thread_pool_device) = vector/norm;
}


void l2_norm_hessian(const ThreadPoolDevice* thread_pool_device, Tensor<type, 1>& vector, Tensor<type, 2>& hessian)
{
    const type norm = l2_norm(thread_pool_device, vector);

    if(norm < type(NUMERIC_LIMITS_MIN))
    {
        hessian.setZero();

        return;
    }

    hessian = self_kronecker_product(thread_pool_device, vector)/(norm*norm*norm);
}


type l2_distance(const Tensor<type, 1>&x, const Tensor<type, 1>&y)
{
    // @todo add thread pool 

    if(x.size() != y.size())
        throw runtime_error("x and y vector must  have the same dimensions.\n");

    Tensor<type, 0> distance;

    distance = (x-y).square().sum().sqrt();

    return distance(0);
}


type l2_distance(const Tensor<type, 2>& x, const Tensor<type, 2>& y)
{
    // @todo add thread pool 

    Tensor<type, 0> distance;

    distance = (x-y).square().sum().sqrt();

    return distance(0);
}


type l2_distance(const type& x, const type& y)
{
    return type(fabs(x - y));
}


Tensor<type, 1> l2_distance(const Tensor<type, 2>& x, const Tensor<type, 2>& y, const Index& size)
{
    // @todo optimize using thread pool and

    Tensor<type, 1> distance(size);

    const Tensor<type, 2> difference = x - y;

    for(Index i = 0; i < difference.dimension(1); i++)
        distance(i) = abs(difference(i));

    return distance;
}


void set_identity(Tensor<type, 2>& matrix)
{
    const Index rows_number = matrix.dimension(0);

    matrix.setZero();

    #pragma omp parallel for

    for(Index i = 0; i < rows_number; i++)
        matrix(i, i) = type(1);
}


void sum_diagonal(Tensor<type, 2>& matrix, const type& value)
{
    const Index rows_number = matrix.dimension(0);

    #pragma omp parallel for

    for(Index i = 0; i < rows_number; i++)
        matrix(i,i) += value;
}


void sum_diagonal(Tensor<type, 2>& matrix, const Tensor<type, 1>& values)
{
    const Index rows_number = matrix.dimension(0);

    #pragma omp parallel for

    for(Index i = 0; i < rows_number; i++)
        matrix(i, i) += values(i);
}


void sum_diagonal(TensorMap<Tensor<type, 2>>& matrix, const Tensor<type, 1>& values)
{
    const Index rows_number = matrix.dimension(0);

    #pragma omp parallel for

    for(Index i = 0; i < rows_number; i++)
        matrix(i, i) += values(i);
}


void substract_diagonal(Tensor<type, 2>& matrix, const Tensor<type, 1>& values)
{
    const Index rows_number = matrix.dimension(0);

    #pragma omp parallel for

    for(Index i = 0; i < rows_number; i++)
        matrix(i, i) -= values(i);
}


Tensor<type, 1> perform_Householder_QR_decomposition(const Tensor<type, 2>& A, const Tensor<type, 1>& b)
{
    const Index n = A.dimension(0);

    Tensor<type, 1> x(n);

    const Map<Matrix<type, Dynamic, Dynamic>> A_eigen((type*)A.data(), n, n);

    const Map<Matrix<type, Dynamic, 1>> b_eigen((type*)b.data(), n, 1);
    
    Map<Matrix<type, Dynamic, 1>> x_eigen((type*)x.data(), n);

    x_eigen = A_eigen.colPivHouseholderQr().solve(b_eigen);

    return x;
}


void fill_tensor_data(const Tensor<type, 2>& matrix,
                      const Tensor<Index, 1>& rows_indices,
                      const Tensor<Index, 1>& columns_indices,
                      type* tensor_data)
{
    const Index rows_number = rows_indices.size();
    const Index columns_number = columns_indices.size();

    const type* matrix_data = matrix.data();

    #pragma omp parallel for

    for (Index j = 0; j < columns_number; j++)
    {
        const type* matrix_column = matrix_data + matrix.dimension(0) * columns_indices[j];

        type* tensor_value = tensor_data + rows_number * j;

        const type* matrix_value = nullptr;

        const Index* rows_indices_data = rows_indices.data();

        for (Index i = 0; i < rows_number; i++)
        {
            matrix_value = matrix_column + *rows_indices_data;
            rows_indices_data++;
            *tensor_value = *matrix_value;
            tensor_value++;
        }
    }
}


void fill_tensor_data_row_major(const Tensor<type, 2>& matrix,
                                const Tensor<Index, 1>& rows_indices,
                                const Tensor<Index, 1>& columns_indices,
                                type* tensor_data)
{
    const Index rows_number = rows_indices.size();
    const Index columns_number = columns_indices.size();

    const type* matrix_data = matrix.data();

    #pragma omp parallel for

    for(Index i = 0; i < rows_number; i++) 
    {
        const Index row_index = rows_indices(i);

        for(Index j = 0; j < columns_number; j++) 
        {
            const Index column_index = columns_indices(j);
            const type* matrix_value = matrix_data + row_index + matrix.dimension(0) * column_index;
            type* tensor_value = tensor_data + i * columns_number + j;
            *tensor_value = *matrix_value;
        }
    }
}


Index count_NAN(const Tensor<type, 1>& x)
{
    Index count = 0;

    #pragma omp parallel for reduction(+:count)

    for(Index i = 0; i < x.size(); i++)
        if(isnan(x(i))) 
            count++;

    return count;
}


Index count_NAN(const Tensor<type, 2>& x)
{
    Index count = 0;

    #pragma omp parallel for reduction(+: count)

    for(Index i = 0; i < x.size(); i++)
        if(isnan(x(i))) 
            count++;

    return count;
}


Index count_empty(const Tensor<string, 1>& strings)
{
    const Index strings_number = strings.size();

    Index count = 0;

    #pragma omp parallel for reduction(+: count)

    for( Index i = 0; i < strings_number; i++)
    {
        string element = strings(i);

        trim(element);
                
        if(element.empty()) 
            count++;
    }

    return count;
}


Index count_not_empty(const Tensor<string, 1>& strings)
{
    const Index strings_number = strings.size();

    Index count = 0;

    #pragma omp parallel for reduction(+: count)

    for( Index i = 0; i < strings_number; i++)
    {
        string element = strings(i);

        trim(element);

        if(!element.empty()) count++;
    }

    return count;
}


void check_size(const Tensor<type, 1>& vector, const Index& size, const string& log)
{
    if(vector.size() != size)
        throw runtime_error("Size of vector is " + to_string(vector.size()) + ", but must be " + to_string(size) + ".\n");
}


void check_dimensions(const Tensor<type, 2>& matrix, const Index& rows_number, const Index& columns_number, const string& log)
{
    if(matrix.dimension(0) != rows_number)
        throw runtime_error("Number of rows in matrix is " + to_string(matrix.dimension(0)) + ", "
                            "but must be " + to_string(rows_number) + ".\n");

    if(matrix.dimension(1) != columns_number)
        throw runtime_error("Number of columns in matrix is " + to_string(matrix.dimension(0)) + ", "
                            "but must be " + to_string(columns_number) + ".\n");
}


// void check_columns_number(const Tensor<type, 2>& matrix, const Index& columns_number, const string& log)
// {
//     if(matrix.dimension(1) != columns_number)
//         throw runtime_error("Number of columns in matrix is " + to_string(matrix.dimension(0)) + ", "
//                             "but must be " + to_string(columns_number) + ".\n");
// }


// void check_rows_number(const Tensor<type, 2>& matrix, const Index& rows_number, const string& log)
// {
//     if(matrix.dimension(1) != rows_number)
//         throw runtime_error("Number of columns in matrix is " + to_string(matrix.dimension(0)) + ", "
//                             "but must be " + to_string(rows_number) + ".\n");
// }


Tensor<Index, 1> join_vector_vector(const Tensor<Index, 1>& x, const Tensor<Index, 1>& y)
{
    const Index size = x.size() + y.size();

    Tensor<Index, 1> data(size);

    memcpy(data.data(), x.data(), x.size() * sizeof(Index));

    memcpy(data.data() + x.size(), y.data(), y.size() * sizeof(Index));

    return data;
}


Tensor<type, 2> assemble_vector_vector(const Tensor<type, 1>& x, const Tensor<type, 1>& y)
{
    const Index rows_number = x.size();
    const Index columns_number = 2;

    Tensor<type, 2> data(rows_number, columns_number);

    #pragma omp parallel for

    for(Index i = 0; i < rows_number; i++)
    {
        data(i, 0) = x(i);
        data(i, 1) = y(i);
    }

    return data;
}


Tensor<type, 2> assemble_vector_matrix(const Tensor<type, 1>& x, const Tensor<type, 2>& y)
{
    const Index rows_number = x.size();
    const Index columns_number = 1 + y.dimension(1);

    Tensor<type, 2> data(rows_number, columns_number);

    #pragma omp parallel for

    for(Index i = 0; i < rows_number; i++)
    {
        data(i, 0) = x(i);

        for(Index j = 0; j < y.dimension(1); j++)
            data(i, 1+j) = y(i, j);
    }

    return data;
}


Tensor<type, 2> assemble_matrix_vector(const Tensor<type, 2>& x, const Tensor<type, 1>& y)
{
    const Index rows_number = y.size();
    const Index columns_number = x.dimension(1) + 1;

    Tensor<type, 2> data(rows_number, columns_number);

    #pragma omp parallel for

    for(Index i = 0; i < rows_number; i++)
    {
        for(Index j = 0; j < x.dimension(1); j++)
            data(i, j) = x(i, j);

        data(i, columns_number-1) = y(i);
    }

    return data;
}


Tensor<type, 2> assemble_matrix_matrix(const Tensor<type, 2>& x, const Tensor<type, 2>& y)
{
    const Index rows_number = x.dimension(0);
    const Index columns_number = x.dimension(1) + y.dimension(1);

    Tensor<type, 2> data(rows_number, columns_number);

    #pragma omp parallel for

    for(Index i = 0; i < rows_number; i++)
    {
        for(Index j = 0; j < x.dimension(1); j++)
            data(i, j) = x(i, j);

        for(Index j = 0; j < y.dimension(1); j++)
            data(i, x.dimension(1) + j) = y(i, j);
    }

    return data;
}


Tensor<string, 1> assemble_text_vector_vector(const Tensor<string, 1>& x, const Tensor<string, 1>& y)
{
    const Index x_size = x.size();
    const Index y_size = y.size();

    Tensor<string,1> data(x_size + y_size);

    #pragma omp parallel for

    for(Index i = 0; i < x_size; i++)
        data(i) = x(i);

    #pragma omp parallel for

    for(Index i = 0; i < y_size; i++)
        data(i + x_size) = y(i);

    return data;
}


string dimensions_to_string(const dimensions& x, const string& separator)
{
    const Index size = x.size();

    ostringstream buffer;

    if(x.size() == 0)
        throw runtime_error("Error: Dimensions size must be greater than 0.\n");

    for(Index i = 0; i < size; i++)
        buffer << x[i] << separator;

    return buffer.str();
}


string tensor_to_string(const Tensor<type, 1>& x, const string& separator)
{
    const Index size = x.size();

    ostringstream buffer;

    if(x.size() == 0)
        throw runtime_error("Error: Dimensions size must be greater than 0.\n");

    for(Index i = 0; i < size; i++)
        buffer << x[i] << separator;

    return buffer.str();
}


string tensor_to_string(const Tensor<Index, 1>& x, const string& separator)
{
    const Index size = x.size();

    ostringstream buffer;

    for(Index i = 0; i < size; i++)
        buffer << x[i] << separator;

    return buffer.str();
}


string string_tensor_to_string(const Tensor<string,1>& x, const string& separator)
{
    const Index size = x.size();

    if(x.size() == 0)
       throw runtime_error("Input vector must have dimension greater than 0.\n");

    string line = x(0);

    for(Index i = 1; i < size; i++)
        line = line + separator + x(i);

    return line;
}


Tensor<type, 2> delete_row(const Tensor<type, 2>& tensor, const Index& row_index)
{
    const Index rows_number = tensor.dimension(0);
    const Index columns_number = tensor.dimension(1);

   #ifdef OPENNN_DEBUG
   if(row_index > rows_number)
      throw runtime_error("row index: " + to_string(row_index) + "rows_number" + to_string(rows_number) + "\n");
   else if(rows_number < 2)
      throw runtime_error("Number of rows must be equal or greater than two.\n");
   #endif

   Tensor<type, 2> new_matrix(rows_number-1, columns_number);

    #pragma omp parallel for

   for(Index i = 0; i < row_index; i++)
      for(Index j = 0; j < columns_number; j++)
          new_matrix(i, j) = tensor(i, j);

    #pragma omp parallel for

   for(Index i = row_index+1; i < rows_number; i++)
      for(Index j = 0; j < columns_number; j++)
         new_matrix(i-1,j) = tensor(i, j);

   return new_matrix;
}


bool contains(const Tensor<size_t,1>& vector, const size_t& value)
{
    Tensor<size_t, 1> copy(vector);

    const size_t* it = find(copy.data(), copy.data()+copy.size(), value);

    return it != (copy.data()+copy.size());
}


bool contains(const Tensor<type, 1>& vector, const type& value)
{
    Tensor<type, 1> copy(vector);

    const type* it = find(copy.data(), copy.data()+copy.size(), value);

    return it != (copy.data()+copy.size());
}


bool contains(const Tensor<Index,1>& vector, const Index& value)
{
    Tensor<Index, 1> copy(vector);

    const Index* it = find(copy.data(), copy.data()+copy.size(), value);

    return it != (copy.data()+copy.size());
}


bool contains(const Tensor<string,1>& vector, const string& value)
{
    Tensor<string, 1> copy(vector);

    const string* it = find(copy.data(), copy.data()+copy.size(), value);

    return it != (copy.data()+copy.size());
}


void push_back_index(Tensor<Index, 1>& old_vector, const Index& new_element)
{
    const Index old_size = old_vector.size();

    const Index new_size = old_size+1;

    Tensor<Index, 1> new_vector(new_size);

    #pragma omp parallel for

    for(Index i = 0; i < old_size; i++) 
        new_vector(i) = old_vector(i);

    new_vector(new_size-1) = new_element;

    old_vector = new_vector;
}


void push_back_string(Tensor<string, 1>& old_vector, const string& new_string)
{
    const Index old_size = old_vector.size();

    const Index new_size = old_size+1;

    Tensor<string, 1> new_vector(new_size);

    #pragma omp parallel for

    for(Index i = 0; i < old_size; i++) 
        new_vector(i) = old_vector(i);

    new_vector(new_size-1) = new_string;

    old_vector = new_vector;
}


void push_back_type(Tensor<type, 1>& vector, const type& new_value)
{
    const Index old_size = vector.size();

    const Index new_size = old_size+1;

    Tensor<type, 1> new_vector(new_size);

    #pragma omp parallel for

    for(Index i = 0; i < old_size; i++) 
        new_vector(i) = vector(i);

    new_vector(new_size-1) = new_value;

    vector = new_vector;
}


Tensor<string, 1> to_string_tensor(const Tensor<type, 1>& x)
{
    Tensor<string, 1> vector(x.size());

    #pragma omp parallel for

    for(Index i = 0; i < x.size(); i++)
        vector(i) = to_string(x(i));

    return vector;
}


void swap_rows(Tensor<type, 2>& matrix, const Index& row_1, const Index& row_2)
{
    const Tensor<type, 1> row = matrix.chip(row_1, 0);

    matrix.chip(row_1, 0) = matrix.chip(row_2, 0);

    matrix.chip(row_2, 0) = row;
}


// Index partition(Tensor<type, 2>& data_matrix,
//                 const Index& start_index,
//                 const Index& end_index,
//                 const Index& target_column)
// {
//     const Tensor<type, 1> pivot_row = data_matrix.chip(start_index, 0);
//     const type pivot_value = pivot_row(target_column);
//     Index smaller_elements_count = 0;

//     for(Index current_index = start_index + 1; current_index <= end_index; current_index++)
//     {
//         if(data_matrix(current_index, target_column) <= pivot_value)
//         {
//             smaller_elements_count++;
//         }
//     }

//     Index pivot_position = start_index + smaller_elements_count;
//     swap_rows(data_matrix, pivot_position, start_index);

//     Index left_index = start_index;
//     Index right_index = end_index;

//     while(left_index < pivot_position && right_index > pivot_position)
//     {
//         while(data_matrix(left_index, target_column) <= pivot_value)
//         {
//             left_index++;
//         }

//         while(data_matrix(right_index, target_column) > pivot_value)
//         {
//             right_index--;
//         }

//         if(left_index < pivot_position && right_index > pivot_position)
//         {
//             swap_rows(data_matrix, left_index++, right_index--);
//         }
//     }

//     return pivot_position;
// }


Tensor<Index, 1> intersection(const Tensor<Index, 1>& tensor_1, const Tensor<Index, 1>& tensor_2)
{
    Index intersection_index_number = 0;

    for(Index i = 0; i < tensor_1.size(); i++)
        for(Index j = 0; j < tensor_2.size(); j++)
            if(tensor_1(i) == tensor_2(j))
                intersection_index_number++;

    if(intersection_index_number == 0)
        return Tensor<Index, 1>(0);

    Tensor<Index, 1> intersection(intersection_index_number);
    Index count = 0;

    for(Index i = 0; i < tensor_1.size(); i++)
        for(Index j = 0; j < tensor_2.size(); j++)
            if(tensor_1(i) == tensor_2(j))
                intersection(count++) = tensor_2(j);

    return intersection;
}


type round_to_precision(type x, const int& precision)
{
    const type factor = type(pow(10, precision));

    return round(factor*x)/factor;
}


// Tensor<type,2> round_to_precision_matrix(Tensor<type,2> matrix,const int& precision)
// {
//     Tensor<type, 2> matrix_rounded(matrix.dimension(0), matrix.dimension(1));

//     const type factor = type(pow(10, precision));

//     for(int i = 0; i < matrix.dimension(0); i++)
//     {
//         for(int j = 0; j < matrix.dimension(1); j++)
//         {
//             matrix_rounded(i, j) = (round(factor*matrix(i, j)))/factor;
//         }
//     }

//     return matrix_rounded;
// }


// Tensor<type, 1> round_to_precision_tensor(Tensor<type, 1> tensor, const int& precision)
// {
//     Tensor<type, 1> tensor_rounded(tensor.size());

//     const type factor = type(pow(10, precision));

//     for(Index i = 0; i < tensor.size(); i++)
//     {
//         tensor_rounded(i) = round(factor*tensor(i))/factor;
//     }

//     return tensor_rounded;
// }


TensorMap<Tensor<type, 1>> tensor_map(const Tensor<type, 2>& matrix, const Index& column_index)
{
    return TensorMap<Tensor<type, 1>>((type*) matrix.data() + column_index * matrix.dimension(0),
                                      matrix.dimension(0));
}


void print_dimensions(const dimensions& new_dimensions)
{
    for(size_t i = 0; i < new_dimensions.size(); i++)
        cout << new_dimensions[i] << " ";

    cout << endl;
}


TensorMap<Tensor<type, 1>> tensor_map_1(const pair<type*, dimensions>& x_pair)
{
    return TensorMap<Tensor<type, 1>>(x_pair.first,
                                      x_pair.second[0]);
}


TensorMap<Tensor<type, 2>> tensor_map_2(const pair<type*, dimensions>& x_pair)
{
    return TensorMap<Tensor<type, 2>>(x_pair.first,
                                      x_pair.second[0],
                                      x_pair.second[1]);
}


TensorMap<Tensor<type, 3>> tensor_map_3(const pair<type*, dimensions>& x_pair)
{
    return TensorMap<Tensor<type, 3>>(x_pair.first,
                                      x_pair.second[0],
                                      x_pair.second[1],
                                      x_pair.second[2]);
}


TensorMap<Tensor<type, 4>> tensor_map_4(const pair<type*, dimensions>& x_pair)
{
    return TensorMap<Tensor<type, 4>>(x_pair.first,
                                      x_pair.second[0],
                                      x_pair.second[1],
                                      x_pair.second[2],
                                      x_pair.second[3]);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2024 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
