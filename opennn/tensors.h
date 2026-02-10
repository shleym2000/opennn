//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E N S O R   U T I L I T I E S   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "pch.h"

namespace opennn
{

struct Shape {
    static constexpr size_t MaxRank = 8;
    Index dims[MaxRank] = {0};
    size_t rank = 0;

    // Constructores...
    Shape() noexcept = default;
    Shape(std::initializer_list<Index> list) {
        rank = std::min(list.size(), MaxRank);
        size_t i = 0;
        for (Index d : list) if (i < rank) dims[i++] = d;
    }

    Shape(size_t n, Index value) {
        rank = (n > MaxRank) ? MaxRank : n;
        for (size_t i = 0; i < rank; ++i) {
            dims[i] = value;
        }
    }

    // --- ACCESO DIRECTO ---

    // Esto permite hacer: Index n = my_shape[0];
    const Index& operator[](size_t i) const {
        // assert(i < rank); // Opcional: para depuración
        return dims[i];
    }

    // Esto permite hacer: my_shape[0] = 128;
    Index& operator[](size_t i) {
        // assert(i < rank); // Opcional: para depuración
        return dims[i];
    }

    // El método back() que añadimos antes
    Index& back() { return dims[rank - 1]; }
    const Index& back() const { return dims[rank - 1]; }

    // --- MÉTODOS DE COMPATIBILIDAD ---
    size_t size() const noexcept { return rank; }
    bool empty() const noexcept { return rank == 0; }
    Index* begin() noexcept { return dims; }
    Index* end() noexcept { return dims + rank; }
    const Index* begin() const noexcept { return dims; }
    const Index* end() const noexcept { return dims + rank; }

    void push_back(Index d) {
        if (rank < MaxRank) dims[rank++] = d;
    }


    void insert(const Index* /*pos*/, const Index* first, const Index* last) {
        while (first != last) {
            this->push_back(*first);
            ++first;
        }
    }


    Index count() const noexcept {
        if (rank == 0) return 0;
        Index total = 1;
        for (size_t i = 0; i < rank; ++i) {
            total *= dims[i];
        }
        return total;
    }

    void clear() noexcept {
        rank = 0;
    }


    void resize(size_t n) {
        if (n > MaxRank) {
            // Esto es crítico para no corromper la pila (stack)
            throw std::out_of_range("Shape::resize: rank exceeds MaxRank (8)");
        }
        rank = n;
    }


    void resize(size_t n, Index value) {
        resize(n);
        for (size_t i = 0; i < rank; ++i) {
            dims[i] = value;
        }
    }

    friend ostream& operator<<(ostream& os, const Shape& s) {
        os << "[ ";
        for (size_t i = 0; i < s.rank; ++i) {
            os << s.dims[i] << (i < s.rank - 1 ? ", " : " ");
        }
        os << "]";
        return os;
    }
};


struct TensorView
{
    type* data = nullptr;
    Shape dims;

    TensorView() noexcept = default;

    TensorView(type* new_data, const Shape& new_shape) noexcept
    {
        data = new_data;
        dims = new_shape;
    }

    Index rank() const { return dims.size(); }

    Index size() const
    {
        if (dims.empty()) return 0;

        return accumulate(dims.begin(), dims.end(), static_cast<Index>(1), multiplies<Index>());
    }

    void print() const
    {
        if(!data || dims.empty())
        {
            cout << "TensorView: Empty or Null" << endl;
            return;
        }

        cout << "Dims: (";
        for(size_t i = 0; i < dims.size(); ++i)
            cout << dims[i] << (i < dims.size() - 1 ? ", " : "");

        cout << ")" << endl;

        const Index total_size = size();
        const Index last_dim_stride = dims.back();

        for(Index i = 0; i < total_size; ++i)
        {
            cout << data[i] << " ";

            if (dims.size() > 1 && (i + 1) % last_dim_stride == 0)
                cout << endl;
        }

        if (dims.size() == 1 || total_size % last_dim_stride != 0)
            cout << endl;
    }
};


type* link(type*, vector<TensorView*>);
void link(type*, vector<vector<TensorView*>>);

Index get_size(const vector<TensorView*>);
Index get_size(vector<vector<TensorView*>>);


template<typename T, size_t N>
using array = Eigen::array<T, N>;

template <typename Index>
Eigen::array<IndexPair<Index>, 1> axes(const Index a, Index b)
{
    return Eigen::array<IndexPair<Index>, 1>({IndexPair<Index>(a, b)});
}


template <typename Index>
Eigen::array<IndexPair<Index>, 2> axes(const Index a1, Index b1, Index a2, Index b2)
{
    return Eigen::array<IndexPair<Index>, 2>({IndexPair<Index>(a1, b1), IndexPair<Index>(a2, b2)});
}


inline Eigen::array<Index, 1> array_1(const Index a)
{
    return Eigen::array<Index, 1>({a});
}


inline Eigen::array<Index, 2> array_2(const Index a, Index b)
{
    return Eigen::array<Index, 2>({a, b});
}


inline Eigen::array<Index, 3> array_3(const Index a, Index b, Index c)
{
    return Eigen::array<Index, 3>({a, b, c});
}


inline Eigen::array<Index, 4> array_4(const Index a, Index b, Index c, Index d)
{
    return Eigen::array<Index, 4>({a, b, c, d});
}


inline array<Index, 5> array_5(const Index a, Index b, Index c, Index d, Index e)
{
    return array<Index, 5>({a, b, c, d, e});
}

type bound(const type value, const type& minimum, const type& maximum);

void set_row(Tensor2&, const Tensor1&, Index);

void sum_matrices(const ThreadPoolDevice*, const Tensor1&, Tensor3&);

void multiply_matrices(const ThreadPoolDevice*, Tensor3&, const Tensor1&);
void multiply_matrices(const ThreadPoolDevice*, Tensor3&, const Tensor2&);

void set_identity(Tensor2&);

void sum_diagonal(Tensor2&, const type&);

Tensor2 self_kronecker_product(const ThreadPoolDevice*, const Tensor1&);

void divide_columns(const ThreadPoolDevice*, TensorMap2, const Tensor1&);

template <int Rank>
bool is_binary(const TensorR<Rank>& tensor)
{
    const Index size = tensor.size();

    for(Index i = 0; i < size; i++)
        if (tensor(i) != type(0) && tensor(i) != type(1) && !isnan(tensor(i)))
            return false;

    return true;
}

Tensor2 append_rows(const Tensor2& , const Tensor2& );

template<typename T>
vector<T> gather_by_index(const vector<T>& data, const vector<Index>& indices)
{
    vector<T> result;
    result.reserve(indices.size());

    for(Index i : indices)
        result.push_back(data[i]);

    return result;
}

vector<Index> build_feasible_rows_mask(const Tensor2& outputs, const Tensor1& minimums, const Tensor1& maximums);

template <int Rank>
bool is_constant(const TensorR<Rank>& tensor)
{
    const Index size = tensor.size();

    Index first_non_nan_index = 0;

    while (first_non_nan_index < size && isnan(tensor(first_non_nan_index)))
        first_non_nan_index++;

    if (first_non_nan_index == size)
        return true;

    const type first_not_nan_element = tensor(first_non_nan_index);

    for(Index i = first_non_nan_index + 1; i < size; ++i)
        if(!isnan(tensor(i)) && abs(first_not_nan_element - tensor(i)) > numeric_limits<float>::min())
            return false;

    return true;
}

void save_csv(const Tensor<type,2>&, const filesystem::path&);

template<int rank>
Index count_NAN(const TensorR<rank>& x)
{
    return count_if(x.data(), x.data() + x.size(), [](type value) {return std::isnan(value); });
}

Index count_between(const Tensor1&, const type&, const type&);

Index count_greater_than(const vector<Index>&, Index);

Tensor<Index, 1> calculate_rank_greater(const Tensor1&);
Tensor<Index, 1> calculate_rank_less(const Tensor1&);

vector<Index> get_elements_greater_than(const vector<Index>&, Index);
vector<Index> get_elements_greater_than(const vector<vector<Index>>&, Index);

Tensor<type,2> filter_column_minimum_maximum(const Tensor<type,2>&, Index, const type&, const type&);

//type l2_distance(const type, const TensorMap<Tensor<type, 0> > &);
type l2_distance(const Tensor1&, const Tensor1&);

Tensor<Index, 1> get_n_nearest_points(const Tensor2& ,const Tensor<type,1>& , int );

void fill_tensor_data_row_major(const Tensor2&, const vector<Index>&, const vector<Index>&, type*);

void fill_tensor_data(const Tensor2&, const vector<Index>&, const vector<Index>&, type*);

void fill_tensor_sequence(const Tensor2&, const vector<Index>&, const vector<Index>&, Index, type*);

template <typename Type, int Rank>
bool contains(const TensorR<Rank>& vector, const Type& value)
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
void push_back(Tensor<T, 1, AlignedMax>& tensor, const T& value)
{
    const int new_size = tensor.dimension(0) + 1;

    Tensor<T, 1> new_tensor(new_size);

    for(int i = 0; i < tensor.dimension(0); i++)
        new_tensor(i) = tensor(i);

    new_tensor(new_size - 1) = value;

    tensor = new_tensor;
}

string dimensions_to_string(const Shape&, const string& = " ");
Shape string_to_dimensions(const string&, const string& = " ");

Shape prepend(Index, const Shape&);

Index get_size(const Shape&);

template <typename T>
string vector_to_string(const vector<T>& x, const string& separator = " ")
{
    ostringstream buffer;

    for(size_t i = 0; i < x.size(); i++)
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
void string_to_tensor(const string& input, Tensor<T, Rank, AlignedMax>& x)
{
    istringstream stream(input);
    T value;
    Index i = 0;

    while (stream >> value)
        x(i++) = value;
}


type round_to_precision(type, const int&);

TensorMap1 tensor_map(const Tensor2&, Index);

TensorMap2 tensor_map(const Tensor3&, Index);
TensorMap3 tensor_map(const Tensor4&, Index);
TensorMap2 tensor_map(const Tensor4&, Index, Index);

TensorMap3 tensor_map_(const TensorMap4&, Index);
//TensorMap1 tensor_map_(const TensorMap2&, Index);

template <Index rank>
TensorMapR<rank> tensor_map(const TensorView& tensor_view)
{
    if(!tensor_view.data)
        throw runtime_error("tensor_map: Null pointer in pair.");

    if (reinterpret_cast<uintptr_t>(tensor_view.data) % EIGEN_MAX_ALIGN_BYTES != 0)
        throw runtime_error("tensor_map alignment error: Pointer is not aligned. "
                            "This will cause a crash with AlignedMax TensorMaps.");

    if constexpr (rank == 2)
        if (tensor_view.rank() == 4)
            return TensorMap2(tensor_view.data,
                              tensor_view.dims[0],
                              tensor_view.size() / tensor_view.dims[0]);

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

    for(size_t i = 0; i < v.size(); i++)
        if (v[i].size() > maximum_size)
            maximum_size = v[i].size();

    return maximum_size;
}


template <typename T>
ostream& operator << (ostream& os, const vector<T>& vec)
{
    os << "[ ";

    for(size_t i = 0; i < vec.size(); ++i)
    {
        os << vec[i];
        if (i + 1 < vec.size())
            os << "; ";
    }

    os << " ]";
    return os;
}



template<class T, int n>
Tensor<Index, 1> get_shape(const Tensor<T, n, AlignedMax>& tensor)
{
    Tensor<Index, 1> shape(n);

    memcpy(shape.data(), tensor.dimensions().data(), size_t(n)*sizeof(Index));

    return shape;
}


template <typename Type, int Rank>
bool is_equal(const Tensor<Type, Rank, AlignedMax>& tensor,
              const Type& value,
              const Type& tolerance = 0.001)
{
    const Index size = tensor.size();

    for(Index i = 0; i < size; i++)
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
bool are_equal(const TensorR<Rank>& tensor_1,
               const TensorR<Rank>& tensor_2,
               const Type& tolerance = 0.001)
{
    if (tensor_1.size() != tensor_2.size())
        throw runtime_error("Tensor sizes are different");

    const Index size = tensor_1.size();

    for(Index i = 0; i < size; i++)
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

struct TensorViewCuda
{
    float* data = nullptr;
    shared_ptr<cudnnTensorStruct> descriptor_handle = nullptr;

    TensorViewCuda() = default;

    TensorViewCuda(float* new_data, std::shared_ptr<cudnnTensorStruct> handle)
        : data(new_data), descriptor_handle(handle) {}

    cudnnTensorDescriptor_t get_descriptor() const 
    {
        return descriptor_handle ? descriptor_handle.get() : nullptr;
    }

    void set_descriptor(const Shape& dims)
    {
        if (descriptor_handle == nullptr)
        {
            cudnnTensorDescriptor_t raw_desc;
            if (cudnnCreateTensorDescriptor(&raw_desc) != CUDNN_STATUS_SUCCESS)
                throw std::runtime_error("TensorViewCuda: Failed to create descriptor.");

            descriptor_handle = std::shared_ptr<cudnnTensorStruct>(raw_desc, [](cudnnTensorDescriptor_t p) {
                if (p) cudnnDestroyTensorDescriptor(p);
                });
        }

        int n = 1, c = 1, h = 1, w = 1;
        if (dims.size() > 0) n = static_cast<int>(dims[0]);
        if (dims.size() > 1) c = static_cast<int>(dims[1]);
        if (dims.size() > 2) h = static_cast<int>(dims[2]);
        if (dims.size() > 3) w = static_cast<int>(dims[3]);

        CHECK_CUDNN(cudnnSetTensor4dDescriptor(descriptor_handle.get(), CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
    }

    Index size() const
    {
        if (descriptor_handle == nullptr) return 0;

        constexpr int REQUESTED_DIMS = CUDNN_DIM_MAX;
        cudnnDataType_t dataType;
        int nbDims = 0, dimA[REQUESTED_DIMS], strideA[REQUESTED_DIMS];

        CHECK_CUDNN(cudnnGetTensorNdDescriptor(descriptor_handle.get(), REQUESTED_DIMS, &dataType, &nbDims, dimA, strideA));

        Index total_elements = 1;
        for (int i = 0; i < nbDims; ++i)
            total_elements *= static_cast<Index>(dimA[i]);
        return total_elements;
    }
};


struct TensorCuda
{
    float* data = nullptr;

    shared_ptr<cudnnTensorStruct> descriptor_handle = nullptr;

    TensorCuda() = default;
    explicit TensorCuda(const Shape& dims) { resize(dims); }

    ~TensorCuda() { if (data) cudaFree(data); }

    TensorCuda(const TensorCuda&) = delete;
    TensorCuda& operator=(const TensorCuda&) = delete;

    TensorCuda(TensorCuda&& other) noexcept
        : data(other.data), descriptor_handle(std::move(other.descriptor_handle))
    {
        other.data = nullptr;
    }

    TensorCuda& operator = (TensorCuda&& other) noexcept
    {
        if (this != &other)
        {
            free();
            data = other.data;
            descriptor_handle = std::move(other.descriptor_handle);
            other.data = nullptr;
        }

        return *this;
    }

    cudnnTensorDescriptor_t get_descriptor() const
    {
        return descriptor_handle ? descriptor_handle.get() : nullptr;
    }

    void resize(const Shape& dims)
    {
        set_descriptor(dims);
        const size_t total_elements = size();
        const size_t bytes = total_elements * sizeof(float);
        if (data) cudaFree(data);
        CHECK_CUDA(cudaMalloc(&data, bytes));
        CHECK_CUDA(cudaMemset(data, 0, bytes));
    }

    void set_descriptor(const Shape& dims)
    {
        if (descriptor_handle == nullptr)
        {
            cudnnTensorDescriptor_t raw_desc;
            if (cudnnCreateTensorDescriptor(&raw_desc) != CUDNN_STATUS_SUCCESS)
                throw std::runtime_error("TensorCuda: Failed to create descriptor.");

            descriptor_handle = std::shared_ptr<cudnnTensorStruct>(raw_desc, [](cudnnTensorDescriptor_t p) {
                if (p) cudnnDestroyTensorDescriptor(p);
                });
        }

        int n = 1, c = 1, h = 1, w = 1;
        if (dims.size() > 0) n = static_cast<int>(dims[0]);
        if (dims.size() > 1) c = static_cast<int>(dims[1]);
        if (dims.size() > 2) h = static_cast<int>(dims[2]);
        if (dims.size() > 3) w = static_cast<int>(dims[3]);

        CHECK_CUDNN(cudnnSetTensor4dDescriptor(descriptor_handle.get(), CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
    }

    Index size() const
    {
        if (descriptor_handle == nullptr) return 0;
        constexpr int REQUESTED_DIMS = CUDNN_DIM_MAX;
        cudnnDataType_t dataType;
        int nbDims = 0, dimA[REQUESTED_DIMS], strideA[REQUESTED_DIMS];

        CHECK_CUDNN(cudnnGetTensorNdDescriptor(descriptor_handle.get(), REQUESTED_DIMS, &dataType, &nbDims, dimA, strideA));

        Index total_elements = 1;
        for (int i = 0; i < nbDims; ++i)
            total_elements *= static_cast<Index>(dimA[i]);
        return total_elements;
    }

    void free()
    {
        if (data) { cudaFree(data); data = nullptr; }
        descriptor_handle.reset();
    }

    TensorViewCuda view() const
    {
        return TensorViewCuda(data, descriptor_handle);
    }
};


type* link(type*, vector<TensorViewCuda*>);
void link(type*, vector<vector<TensorViewCuda*>>);

Index get_size(const vector<TensorViewCuda*>);
Index get_size(vector<vector<TensorViewCuda*>>);

#endif

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
