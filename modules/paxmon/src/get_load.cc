#include "motis/paxmon/get_load.h"
#include "motis/paxmon/get_load_internal.h"

#include <cassert>
#include <algorithm>
#include <numeric>

#ifdef MOTIS_AVX2
#include <immintrin.h>
#endif

#include "utl/enumerate.h"
#include "utl/verify.h"

namespace motis::paxmon {

pax_limits get_pax_limits(pax_connection_info const& pci) {
  auto limits = pax_limits{};
  for (auto const& si : pci.section_infos_) {
    auto const pax = si.group_->passengers_;
    limits.max_ += pax;
    if (si.group_->probability_ == 1.0F) {
      limits.min_ += pax;
    }
  }
  return limits;
}

std::uint16_t get_base_load(pax_connection_info const& pci) {
  std::uint16_t load = 0;
  for (auto const& si : pci.section_infos_) {
    if (si.group_->probability_ == 1.0F) {
      load += si.group_->passengers_;
    }
  }
  return load;
}

std::uint16_t get_expected_load(pax_connection_info const& pci) {
  std::uint16_t load = 0;
  for (auto const& si : pci.section_infos_) {
    if (((si.group_->source_flags_ & group_source_flags::FORECAST) !=
         group_source_flags::FORECAST) &&
        si.group_->probability_ == 1.0F) {
      load += si.group_->passengers_;
    }
  }
  return load;
}

inline void convolve_base(pax_pdf& pdf, std::uint16_t const grp_size,
                          float grp_prob) {
  auto old_pdf = pdf;
  auto const inv_grp_prob = 1.0F - grp_prob;
  for (auto& e : pdf.data_) {
    e *= inv_grp_prob;
  }
  for (auto const& [old_size, old_prob] : utl::enumerate(old_pdf.data_)) {
    if (old_prob == 0.0F) {
      continue;
    }
    auto const added_size = old_size + grp_size;
    utl::verify(pdf.data_.size() > added_size, "convolve: invalid pdf size");
    pdf.data_[added_size] += grp_prob * old_prob;
  }
}

pax_pdf get_load_pdf_base(pax_connection_info const& pci) {
  auto const limits = get_pax_limits(pci);
  auto pdf = pax_pdf{};
  pdf.data_.resize(limits.max_ + 1);
  pdf.data_[limits.min_] = 1.0F;
  for (auto const& si : pci.section_infos_) {
    if (si.group_->probability_ != 1.0F) {
      convolve_base(pdf, si.group_->passengers_, si.group_->probability_);
    }
  }
  return pdf;
}

#ifdef MOTIS_AVX2

template <auto M, typename T>
inline T round_up(T const val) {
  auto const rem = val % M;
  return rem == 0 ? val : val + M - rem;
}

template <auto M, typename T>
inline T round_down(T const val) {
  return val - val % M;
}

inline std::size_t get_start_offset(pax_limits const& limits,
                                    std::uint16_t const grp_size) {
  return round_down<8>(std::max(limits.min_, grp_size));
}

inline void convolve_avx(pax_pdf& pdf, std::uint16_t const grp_size,
                         float grp_prob, pax_limits const& limits,
                         std::vector<float>& buf) {
  assert(pdf.data_.size() % 8 == 0);
  assert(buf.size() == pdf.data_.size() + 8);

  std::copy(begin(pdf.data_), end(pdf.data_), &buf[8]);

  auto const m_grp_prob = _mm256_set1_ps(grp_prob);
  auto const m_inv_grp_prob = _mm256_set1_ps(1.0F - grp_prob);
  for (auto i = 0ULL; i < pdf.data_.size(); i += 8) {
    auto data_ptr = &pdf.data_[i];
    _mm256_storeu_ps(data_ptr,
                     _mm256_mul_ps(_mm256_loadu_ps(data_ptr), m_inv_grp_prob));
  }

  auto const start_offset = get_start_offset(limits, grp_size);
  auto const buf_offset = 8 - grp_size;
  for (auto i = start_offset; i < pdf.data_.size(); i += 8) {
    auto data_ptr = &pdf.data_[i];
    _mm256_storeu_ps(data_ptr,
                     _mm256_fmadd_ps(_mm256_loadu_ps(&buf[i + buf_offset]),
                                     m_grp_prob, _mm256_loadu_ps(data_ptr)));
  }
}

pax_pdf get_load_pdf_avx(pax_connection_info const& pci) {
  auto const limits = get_pax_limits(pci);
  auto pdf = pax_pdf{};
  auto const pdf_size = limits.max_ + 1;
  pdf.data_.resize(round_up<8>(pdf_size));
  pdf.data_[limits.min_] = 1.0F;
  auto buf = std::vector<float>(pdf.data_.size() + 8);

  for (auto const& si : pci.section_infos_) {
    if (si.group_->probability_ != 1.0F) {
      convolve_avx(pdf, si.group_->passengers_, si.group_->probability_, limits,
                   buf);
    }
  }

  pdf.data_.resize(pdf_size);
  return pdf;
}

#endif

pax_pdf get_load_pdf(pax_connection_info const& pci) {
#ifdef MOTIS_AVX2
  return get_load_pdf_avx(pci);
#else
  return get_load_pdf_base(pci);
#endif
}

pax_cdf get_cdf(pax_pdf const& pdf) {
  auto cdf = pax_cdf{};
  cdf.data_.resize(pdf.data_.size());
  auto cumulative_prob = 0.0F;
  for (auto const& [size, prob] : utl::enumerate(pdf.data_)) {
    cumulative_prob += prob;
    cdf.data_[size] = cumulative_prob;
  }
  return cdf;
}

pax_cdf get_load_cdf(pax_connection_info const& pci) {
  return get_cdf(get_load_pdf(pci));
}

template <typename T>
lf_df_t to_load_factor_impl(T const& df, std::uint16_t capacity) {
  auto lf_df = std::map<float, float>{};
  auto last_prob = 0.0F;
  for (auto const& [size, prob] : utl::enumerate(df.data_)) {
    if (prob == 0.0F || prob == last_prob) {
      continue;
    }
    lf_df[static_cast<float>(size) / static_cast<float>(capacity)] = prob;
    last_prob = prob;
  }
  return lf_df;
}

lf_df_t to_load_factor(pax_pdf const& pdf, std::uint16_t capacity) {
  return to_load_factor_impl(pdf, capacity);
}

lf_df_t to_load_factor(pax_cdf const& cdf, std::uint16_t capacity) {
  return to_load_factor_impl(cdf, capacity);
}

void add_additional_group(pax_pdf& pdf, std::uint16_t passengers,
                          float probability) {
  pdf.data_.resize(pdf.data_.size() + passengers);
  convolve_base(pdf, passengers, probability);
}

std::size_t get_max_new_pax(
    std::vector<std::pair<passenger_group const*, float>> const&
        additional_groups) {
  return std::accumulate(
      begin(additional_groups), end(additional_groups), 0ULL,
      [](auto const sum, auto const& p) { return sum + p.first->passengers_; });
}

void add_additional_groups_base(
    pax_pdf& pdf, std::vector<std::pair<passenger_group const*, float>> const&
                      additional_groups) {
  assert(!additional_groups.empty());
  auto const max_new_pax = get_max_new_pax(additional_groups);
  pdf.data_.resize(pdf.data_.size() + max_new_pax);
  for (auto const& [grp, grp_probability] : additional_groups) {
    convolve_base(pdf, grp->passengers_, grp_probability);
  }
}

#ifdef MOTIS_AVX2
void add_additional_groups_avx(
    pax_pdf& pdf, std::vector<std::pair<passenger_group const*, float>> const&
                      additional_groups) {
  assert(!additional_groups.empty());
  auto const max_new_pax = get_max_new_pax(additional_groups);
  auto const pdf_size = pdf.data_.size() + max_new_pax;
  pdf.data_.resize(round_up<8>(pdf_size));
  auto buf = std::vector<float>(pdf.data_.size() + 8);
  auto limits = pax_limits{
      std::min_element(begin(additional_groups), end(additional_groups),
                       [](auto const& p1, auto const& p2) {
                         return p1.first->passengers_ < p2.first->passengers_;
                       })
          ->first->passengers_,
      0};
  for (auto const& [grp, grp_probability] : additional_groups) {
    convolve_avx(pdf, grp->passengers_, grp_probability, limits, buf);
  }
  pdf.data_.resize(pdf_size);
}
#endif

void add_additional_groups(
    pax_pdf& pdf, std::vector<std::pair<passenger_group const*, float>> const&
                      additional_groups) {
  if (additional_groups.empty()) {
    return;
  }
#if MOTIS_AVX2
  return add_additional_groups_avx(pdf, additional_groups);
#else
  return add_additional_groups_base(pdf, additional_groups);
#endif
}

bool load_factor_possibly_ge(pax_pdf const& pdf, std::uint16_t capacity,
                             float threshold) {
  auto const pax_threshold =
      static_cast<std::uint16_t>(static_cast<float>(capacity) * threshold);

  if (pdf.data_.empty() || pax_threshold >= pdf.data_.size()) {
    return false;
  }

  for (auto i = pdf.data_.size() - 1; i >= pax_threshold; i--) {
    if (pdf.data_[i] != 0.0) {
      return true;
    }
  }
  return false;
}

bool load_factor_possibly_ge(pax_cdf const& cdf, std::uint16_t capacity,
                             float threshold) {
  auto const pax_threshold =
      static_cast<std::uint16_t>(static_cast<float>(capacity) * threshold);

  if (cdf.data_.empty() || pax_threshold >= cdf.data_.size()) {
    return false;
  } else {
    return cdf.data_.back() != 0.0;
  }
}

bool load_factor_possibly_ge(lf_df_t const& lf_df, float threshold) {
  return lf_df.lower_bound(threshold) != end(lf_df);
}

}  // namespace motis::paxmon
