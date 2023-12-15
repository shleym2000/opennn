#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <string>
#include "config.h"

using namespace std;
using namespace Eigen;

namespace opennn
{

/// Histograms are visual aid to study the distributions of data set variables.
///
/// This structure contains the essentials for making histograms
/// and obtained the data generated by the histogram :
/// <ul>
/// <li> Minimum.
/// <li> Maximum.
/// <li> Centers.
/// <li> Frequencies.
/// </ul>

struct Histogram
{
  /// Default constructor.

  explicit Histogram();

  /// Bins number constructor.

  explicit Histogram(const Index&);

  /// Values constructor.

  explicit Histogram(const Tensor<type, 1>&, const Tensor<Index, 1>&);

  /// Values constructor 2.

  explicit Histogram(const Tensor<Index, 1>&, const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 1>&);

  /// Data constructor

  explicit Histogram(const Tensor<type, 1>&, const Index&);

  /// Probabillities constructor

  explicit Histogram(const Tensor<type, 1>&);

  // Methods

  Index get_bins_number() const;

  Index count_empty_bins() const;

  Index calculate_minimum_frequency() const;

  Index calculate_maximum_frequency() const;

  Index calculate_most_populated_bin() const;

  Tensor<type, 1> calculate_minimal_centers() const;

  Tensor<type, 1> calculate_maximal_centers() const;

  Index calculate_bin(const type&) const;

  Index calculate_frequency(const type&) const;

  void save(const string&) const;

  /// Minimum positions of the bins in the histogram.

  Tensor<type, 1> minimums;

  /// Maximum positions of the bins in the histogram.

  Tensor<type, 1> maximums;

  /// Positions of the bins in the histogram.

  Tensor<type, 1> centers;

  /// Population of the bins in the histogram.

  Tensor<Index, 1> frequencies;
};


}
#endif
