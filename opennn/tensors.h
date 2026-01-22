#ifndef TENSORS_H
#define TENSORS_H

#include "pch.h"

namespace opennn
{

struct TensorView
{
    type* data = nullptr;
    dimensions dims;

    TensorView() noexcept = default;
    TensorView(type* new_data, dimensions new_dims) noexcept : data(new_data), dims(std::move(new_dims))
    {}

    Index rank() const { return dims.size(); }

    Index size() const
    {
        if (dims.empty()) return 0;

        return accumulate(dims.begin(), dims.end(), static_cast<Index>(1), multiplies<Index>());
    }

    void print() const
    {
        if (!data || dims.empty())
        {
            cout << "TensorView: Empty or Null" << endl;
            return;
        }

        cout << "Dims: (";
        for (size_t i = 0; i < dims.size(); ++i)
            cout << dims[i] << (i < dims.size() - 1 ? ", " : "");

        cout << ")" << endl;

        const Index total_size = size();
        const Index last_dim_stride = dims.back();

        for (Index i = 0; i < total_size; ++i)
        {
            cout << data[i] << " ";

            if (dims.size() > 1 && (i + 1) % last_dim_stride == 0)
                cout << endl;
        }
        if (dims.size() == 1 || total_size % last_dim_stride != 0)
            cout << endl;
    }
};


template<typename T, size_t N>
using array = Eigen::array<T, N>;

template <typename Index>
Eigen::array<IndexPair<Index>, 1> axes(const Index& a, const Index& b)
{
    return Eigen::array<IndexPair<Index>, 1>({IndexPair<Index>(a, b)});
}


template <typename Index>
Eigen::array<IndexPair<Index>, 2> axes(const Index& a1, const Index& b1, const Index& a2, const Index& b2)
{
    return Eigen::array<IndexPair<Index>, 2>({IndexPair<Index>(a1, b1), IndexPair<Index>(a2, b2)});
}


inline Eigen::array<Index, 1> array_1(const Index& a)
{
    return Eigen::array<Index, 1>({a});
}


inline Eigen::array<Index, 2> array_2(const Index& a, const Index& b)
{
    return Eigen::array<Index, 2>({a, b});
}


inline Eigen::array<Index, 3> array_3(const Index& a, const Index& b, const Index& c)
{
    return Eigen::array<Index, 3>({a, b, c});
}


inline Eigen::array<Index, 4> array_4(const Index& a, const Index& b, const Index& c, const Index& d)
{
    return Eigen::array<Index, 4>({a, b, c, d});
}


inline array<Index, 5> array_5(const Index& a, const Index& b, const Index& c, const Index& d, const Index& e)
{
    return array<Index, 5>({a, b, c, d, e});
}

type bound(const type& value, const type& minimum, const type& maximum);

void set_row(Tensor2&, const Tensor1&, const Index&);

void sum_matrices(const ThreadPoolDevice*, const Tensor1&, Tensor3&);

void multiply_matrices(const ThreadPoolDevice*, Tensor3&, const Tensor1&);
void multiply_matrices(const ThreadPoolDevice*, Tensor3&, const Tensor2&);

void set_identity(Tensor2&);

void sum_diagonal(Tensor2&, const type&);

Tensor2 self_kronecker_product(const ThreadPoolDevice*, const Tensor1&);

void divide_columns(const ThreadPoolDevice*, TensorMap2&, const Tensor1&);

template <int Rank>
bool is_binary(const Tensor<type, Rank>& tensor)
{
    const Index size = tensor.size();

    for (Index i = 0; i < size; i++)
        if (tensor(i) != type(0) && tensor(i) != type(1) && !isnan(tensor(i)))
            return false;

    return true;
}

Tensor2 append_rows(const Tensor<type,2>& , const Tensor<type,2>& );

template <int Rank>
bool is_constant(const Tensor<type, Rank>& tensor)
{
    const Index size = tensor.size();

    Index first_non_nan_index = 0;

    while (first_non_nan_index < size && isnan(tensor(first_non_nan_index)))
        first_non_nan_index++;

    if (first_non_nan_index == size)
        return true;

    const type first_not_nan_element = tensor(first_non_nan_index);

    for (Index i = first_non_nan_index + 1; i < size; ++i)
        if (!isnan(tensor(i)) && abs(first_not_nan_element - tensor(i)) > numeric_limits<float>::min())
            return false;

    return true;
}

void save_csv(const Tensor<type,2>&, const filesystem::path&);

template<int rank>
Index count_NAN(const Tensor<type, rank>& x)
{
    return count_if(x.data(), x.data() + x.size(), [](type value) {return std::isnan(value); });
}

Index count_between(Tensor1&, const type&, const type&);

Index count_greater_than(const vector<Index>&, const Index&);

Tensor<Index, 1> calculate_rank_greater(const Tensor1&);
Tensor<Index, 1> calculate_rank_less(const Tensor1&);

vector<Index> get_elements_greater_than(const vector<Index>&, const Index&);
vector<Index> get_elements_greater_than(const vector<vector<Index>>&, const Index&);

Tensor<type,2> filter_column_minimum_maximum(const Tensor<type,2>&, const Index&, const type&, const type&);

//type l2_distance(const type&, const TensorMap<Tensor<type, 0> > &);
type l2_distance(const Tensor1&, const Tensor1&);

Tensor<Index, 1> get_n_nearest_points(const Tensor2& ,const Tensor<type,1>& , int );

void fill_tensor_data_row_major(const Tensor2&, const vector<Index>&, const vector<Index>&, type*);

void fill_tensor_data(const Tensor2&, const vector<Index>&, const vector<Index>&, type*);

void fill_tensor_sequence(const Tensor2&, const vector<Index>&, const vector<Index>&, const Index&, type*);

template <typename Type, int Rank>
bool contains(const Tensor<Type, Rank>& vector, const Type& value)
{
    Tensor<Type, 1> copy(vector);

    const Type* it = find(copy.data(), copy.data() + copy.size(), value);

    return it != (copy.data() + copy.size());
}


bool contains(const vector<string>&, const string&);

Tensor1 perform_Householder_QR_decomposition(const Tensor2&, const Tensor1&);

vector<Index> join_vector_vector(const vector<Index>&, const vector<Index>&);

Tensor2 assemble_vector_vector(const Tensor1&, const Tensor1&);
Tensor2 assemble_vector_matrix(const Tensor1&, const Tensor2&);
Tensor2 assemble_matrix_matrix(const Tensor2&, const Tensor2&);

template <typename T>
void push_back(Tensor<T, 1>& tensor, const T& value)
{
    const int new_size = tensor.dimension(0) + 1;

    Tensor<T, 1> new_tensor(new_size);

    for (int i = 0; i < tensor.dimension(0); i++)
        new_tensor(i) = tensor(i);

    new_tensor(new_size - 1) = value;

    tensor = new_tensor;
}

string dimensions_to_string(const dimensions&, const string& = " ");
dimensions string_to_dimensions(const string&, const string& = " ");

dimensions prepend(const Index& x, const dimensions& d);

Index get_size(const dimensions& d);

template <typename T>
string vector_to_string(const vector<T>& x, const string& separator = " ")
{
    ostringstream buffer;

    for (size_t i = 0; i < x.size(); i++)
    {
        buffer << x[i];
        if (i < x.size() - 1)
            buffer << separator;
    }

    return buffer.str();
}


template <typename T, size_t Rank>
string tensor_to_string(const Tensor<T, Rank>& x, const string& separator = " ")
{
    ostringstream buffer;

    for(Index i = 0; i < x.size(); i++)
        buffer << x(i) << separator;

    return buffer.str();
}


template <typename T, size_t Rank>
void string_to_tensor(const string& input, Tensor<T, Rank>& x)
{
    istringstream stream(input);
    T value;
    Index i = 0;

    while (stream >> value)
        x(i++) = value;
}


type round_to_precision(type, const int&);

TensorMap1 tensor_map(const Tensor2&, const Index&);

TensorMap2 tensor_map(const Tensor3&, const Index&);
TensorMap3 tensor_map(const Tensor4&, const Index&);
TensorMap2 tensor_map(const Tensor4&, const Index&, const Index&);

TensorMap3 tensor_map_(const TensorMap4&, const Index&);
//TensorMap1 tensor_map_(const TensorMap2&, const Index&);

template <Index rank>
TensorMap<Tensor<type, rank>, Aligned16> tensor_map(const TensorView& tensor_view)
{
    if (!tensor_view.data)
        throw runtime_error("tensor_map: Null pointer in pair.");

    if (reinterpret_cast<uintptr_t>(tensor_view.data) % 16 != 0)
        throw runtime_error("tensor_map alignment error: Pointer is not 16-byte aligned. "
                            "This will cause a crash with Aligned16 TensorMaps.");

    if (tensor_view.rank() != rank)
        throw runtime_error("Dimensions is " + to_string(tensor_view.rank()) + " and must be " + to_string(rank));

    if constexpr (rank == 1)
        return TensorMap1(tensor_view.data, tensor_view.dims[0]);
    else if constexpr (rank == 2)
        return TensorMap2(tensor_view.data,
                          tensor_view.dims[0],
                          tensor_view.dims[1]);
    else if constexpr (rank == 3)
        return TensorMap3(tensor_view.data,
                          tensor_view.dims[0],
                          tensor_view.dims[1],
                          tensor_view.dims[2]);
    else if constexpr (rank == 4)
        return TensorMap4(tensor_view.data,
                          tensor_view.dims[0],
                          tensor_view.dims[1],
                          tensor_view.dims[2],
                          tensor_view.dims[3]);
    else
        static_assert(rank >= 1 && rank <= 4, "Unsupported tensor rank");
}


template <typename T>
size_t get_maximum_size(const vector<vector<T>>& v)
{
    size_t maximum_size = 0;

    for (size_t i = 0; i < v.size(); i++)
        if (v[i].size() > maximum_size)
            maximum_size = v[i].size();

    return maximum_size;
}


template <typename T>
ostream& operator << (ostream& os, const vector<T>& vec)
{
    os << "[ ";

    for (size_t i = 0; i < vec.size(); ++i)
    {
        os << vec[i];
        if (i + 1 < vec.size())
            os << "; ";
    }

    os << " ]";
    return os;
}



template<class T, int n>
Tensor<Index, 1> get_dimensions(const Tensor<T, n>& tensor)
{
    Tensor<Index, 1> dimensions(n);

    memcpy(dimensions.data(), tensor.dimensions().data(), size_t(n)*sizeof(Index));

    return dimensions;
}


template <typename Type, int Rank>
bool is_equal(const Tensor<Type, Rank>& tensor,
              const Type& value,
              const Type& tolerance = 0.001)
{
    const Index size = tensor.size();

    for (Index i = 0; i < size; i++)
        if constexpr (is_same_v<Type, bool>)
        {
            if (tensor(i) != value)
                return false;
            else
                if (std::abs(tensor(i) - value) > tolerance)
                    return false;
        }

    return true;
}


template <typename Type, int Rank>
bool are_equal(const Tensor<Type, Rank>& tensor_1,
               const Tensor<Type, Rank>& tensor_2,
               const Type& tolerance = 0.001)
{
    if (tensor_1.size() != tensor_2.size())
        throw runtime_error("Tensor sizes are different");

    const Index size = tensor_1.size();

    for (Index i = 0; i < size; i++)
        if constexpr (is_same_v<Type, bool>)
        {
            if (tensor_1(i) != tensor_2(i))
                return false;
            else if (abs(tensor_1(i) - tensor_2(i)) > tolerance)
                return false;
        }

    return true;
}

#ifdef OPENNN_CUDA

struct TensorCuda
{
    float* data = nullptr;
    cudnnTensorDescriptor_t descriptor = nullptr;

    TensorCuda() = default;

    TensorCuda(float* new_data, cudnnTensorDescriptor_t new_descriptor)
        : data(new_data), descriptor(new_descriptor) {}

    Index size() const
    {
        throw runtime_error ("Not implemented yet");
        // use descriptor to return size.

        return 0;
    }
};



struct TensorViewCuda
{
    float* data = nullptr;
    cudnnTensorDescriptor_t descriptor = nullptr;

    TensorViewCuda() = default;

    TensorViewCuda(float* new_data, cudnnTensorDescriptor_t new_descriptor)
        : data(new_data), descriptor(new_descriptor) {}

    ~TensorViewCuda()
    {
        data = nullptr;

        if (descriptor)
        {
            cudnnDestroyTensorDescriptor(descriptor);
            descriptor = nullptr;
        }
    }
/*
    TensorViewCuda(const TensorViewCuda&) = delete;

    TensorViewCuda& operator=(const TensorViewCuda&) = delete;

    TensorViewCuda(TensorViewCuda&& other) noexcept
        : data(other.data), descriptor(other.descriptor)
    {
        other.data = nullptr;
        other.descriptor = nullptr;
    }

    TensorViewCuda& operator=(TensorViewCuda&& other) noexcept
    {
        if (this != &other)
        {
            if (descriptor) cudnnDestroyTensorDescriptor(descriptor);

            data = other.data;
            descriptor = other.descriptor;

            other.data = nullptr;
            other.descriptor = nullptr;
        }
        return *this;
    }
*/
    void set_descriptor(const dimensions& dims)
    {
        // 1. Create the descriptor if it doesn't exist yet
        if (descriptor == nullptr)
        {
            if (cudnnCreateTensorDescriptor(&descriptor) != CUDNN_STATUS_SUCCESS)
                throw runtime_error("TensorViewCuda: Failed to create descriptor.");
        }

        // 2. Map variable dimensions to 4D (N, C, H, W)
        // Defaults are 1. This handles cases like 2D Dense tensors [Batch, Size]
        // becoming [Batch, Size, 1, 1].
        int n = 1, c = 1, h = 1, w = 1;

        if (dims.size() > 0) n = static_cast<int>(dims[0]);
        if (dims.size() > 1) c = static_cast<int>(dims[1]);
        if (dims.size() > 2) h = static_cast<int>(dims[2]);
        if (dims.size() > 3) w = static_cast<int>(dims[3]);

        // 3. Configure the descriptor
        // Assuming Standard Layout (NCHW) and Float type based on your codebase
        cudnnStatus_t status = cudnnSetTensor4dDescriptor(
            descriptor,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            n, c, h, w
            );

        if (status != CUDNN_STATUS_SUCCESS)
            throw runtime_error("TensorViewCuda: Failed to set 4D descriptor.");
    }

    Index size() const
    {
        if (descriptor == nullptr)
            throw runtime_error("TensorViewCuda::size(): Descriptor is nullptr. Cannot calculate size.");

        constexpr int REQUESTED_DIMS = CUDNN_DIM_MAX;

        cudnnDataType_t dataType;
        int nbDims = 0;
        int dimA[REQUESTED_DIMS];
        int strideA[REQUESTED_DIMS];

        cudnnStatus_t status = cudnnGetTensorNdDescriptor(
            descriptor,
            REQUESTED_DIMS,
            &dataType,
            &nbDims,
            dimA,
            strideA
        );

        if (status != CUDNN_STATUS_SUCCESS)
            throw runtime_error(string("TensorViewCuda::size(): Failed to get descriptor info. Error: ") + cudnnGetErrorString(status));

        Index total_elements = 1;
        for (int i = 0; i < nbDims; ++i)
            total_elements *= static_cast<Index>(dimA[i]);

        return total_elements;
    }
};

#endif

}

#endif
