#pragma once

#include <cassert>
#include <algorithm>
#include <limits>
#include <map>
#include <numeric>
#include <utility>
#include <vector>

#ifdef MOTIS_AVX2
#include <immintrin.h>
#endif

#include "utl/enumerate.h"
#include "utl/verify.h"

#include "cista/reflection/comparable.h"

#include "motis/paxmon/additional_group.h"
#include "motis/paxmon/passenger_group.h"
#include "motis/paxmon/passenger_group_container.h"
#include "motis/paxmon/pci_container.h"
#include "motis/paxmon/universe.h"

namespace motis::paxmon {

struct pax_limits {
  CISTA_COMPARABLE()

  std::uint16_t min_{};
  std::uint16_t max_{};
};

struct pax_pdf {
  CISTA_COMPARABLE()

  std::vector<float> data_;
};

struct pax_cdf {
  CISTA_COMPARABLE()

  std::vector<float> data_;
};

struct pax_stats {
  pax_limits limits_{};
  std::uint16_t q5_{};
  std::uint16_t q50_{};
  std::uint16_t q95_{};
};

using lf_df_t = std::map<float, float>;

auto const constexpr INVALID_PGI =
    std::numeric_limits<passenger_group_index>::max();

template <typename GroupRoutes>
inline pax_limits get_pax_limits(passenger_group_container const& pgc,
                                 GroupRoutes const& group_routes) {
  auto limits = pax_limits{};
  auto current_pgi = INVALID_PGI;
  auto current_prob = 0.0F;

  for (auto const& pgwr : group_routes) {
    auto const prob = pgc.route(pgwr).probability_;
    if (prob == 0.0F) {
      continue;
    }
    if (current_pgi != pgwr.pg_) {
      if (current_pgi != INVALID_PGI /*&& current_prob != 0.0F*/) {
        auto const current_pax = pgc[current_pgi]->passengers_;
        limits.max_ += current_pax;
        if (current_prob == 1.0F) {
          limits.min_ += current_pax;
        }
      }
      current_pgi = pgwr.pg_;
      current_prob = prob;
    } else {
      current_prob += prob;
    }
  }

  if (current_pgi != INVALID_PGI /*&& current_prob != 0.0F*/) {
    auto const current_pax = pgc[current_pgi]->passengers_;
    limits.max_ += current_pax;
    if (current_prob == 1.0F) {
      limits.min_ += current_pax;
    }
  }

  return limits;
}

template <typename GroupRoutes>
inline std::uint16_t get_base_load(passenger_group_container const& pgc,
                                   GroupRoutes const& group_routes) {
  std::uint16_t load = 0;
  for (auto const& pgwr : group_routes) {
    if (pgc.route(pgwr).probability_ == 1.0F) {
      load += pgc[pgwr.pg_]->passengers_;
    }
  }
  return load;
}

std::uint16_t get_expected_load(pci_container const& pcis, pci_index idx);
std::uint16_t get_expected_load(universe const& uv, pci_index idx);

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

template <typename GroupRoutes>
inline pax_pdf get_load_pdf_base(passenger_group_container const& pgc,
                                 GroupRoutes const& group_routes) {
  auto const limits = get_pax_limits(pgc, group_routes);
  auto pdf = pax_pdf{};
  pdf.data_.resize(limits.max_ + 1);
  pdf.data_[limits.min_] = 1.0F;

  auto current_pgi = INVALID_PGI;
  auto current_prob = 0.0F;

  for (auto const& pgwr : group_routes) {
    auto const prob = pgc.route(pgwr).probability_;
    if (prob != 1.0F && prob != 0.0F) {
      if (current_pgi != pgwr.pg_) {
        if (current_pgi != INVALID_PGI && current_prob != 0.0F &&
            current_prob != 1.0F) {
          convolve_base(pdf, pgc[current_pgi]->passengers_, current_prob);
        }
        current_pgi = pgwr.pg_;
        current_prob = prob;
      } else {
        current_prob += prob;
      }
    }
  }

  if (current_pgi != INVALID_PGI && current_prob != 0.0F &&
      current_prob != 1.0F) {
    convolve_base(pdf, pgc[current_pgi]->passengers_, current_prob);
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

template <typename GroupRoutes>
inline pax_pdf get_load_pdf_avx(passenger_group_container const& pgc,
                                GroupRoutes const& group_routes) {
  auto const limits = get_pax_limits(pgc, group_routes);
  auto pdf = pax_pdf{};
  auto const pdf_size = limits.max_ + 1;
  pdf.data_.resize(round_up<8>(pdf_size));
  pdf.data_[limits.min_] = 1.0F;
  auto buf = std::vector<float>(pdf.data_.size() + 8);

  auto current_pgi = INVALID_PGI;
  auto current_prob = 0.0F;

  for (auto const& pgwr : group_routes) {
    auto const prob = pgc.route(pgwr).probability_;
    if (prob != 1.0F && prob != 0.0F) {
      if (current_pgi != pgwr.pg_) {
        if (current_pgi != INVALID_PGI && current_prob != 0.0F &&
            current_prob != 1.0F) {
          convolve_avx(pdf, pgc[current_pgi]->passengers_, current_prob, limits,
                       buf);
        }
        current_pgi = pgwr.pg_;
        current_prob = prob;
      } else {
        current_prob += prob;
      }
    }
  }

  if (current_pgi != INVALID_PGI && current_prob != 0.0F &&
      current_prob != 1.0F) {
    convolve_avx(pdf, pgc[current_pgi]->passengers_, current_prob, limits, buf);
  }

  pdf.data_.resize(pdf_size);
  return pdf;
}

#endif

template <typename GroupRoutes>
inline pax_pdf get_load_pdf(passenger_group_container const& pgc,
                            GroupRoutes const& group_routes) {
#ifdef MOTIS_AVX2
  return get_load_pdf_avx(pgc, group_routes);
#else
  return get_load_pdf_base(pgc, group_routes);
#endif
}

template <typename GroupRoutes>
inline pax_cdf get_load_cdf(passenger_group_container const& pgc,
                            GroupRoutes const& group_routes) {
  return get_cdf(get_load_pdf(pgc, group_routes));
}

pax_cdf get_cdf(pax_pdf const& pdf);

template <typename GroupRoutes>
inline std::uint16_t get_mean_load(passenger_group_container const& pgc,
                                   GroupRoutes const& group_routes) {
  if (group_routes.empty()) {
    return 0;
  }
  auto mean = 0.0F;
  for (auto const& pgwr : group_routes) {
    mean += static_cast<float>(pgc[pgwr.pg_]->passengers_) *
            pgc.route(pgwr).probability_;
  }
  return static_cast<std::uint16_t>(mean);
}

std::uint16_t get_pax_quantile(pax_cdf const& cdf, float q);
std::uint16_t get_median_load(pax_cdf const& cdf);
pax_stats get_pax_stats(pax_cdf const& cdf);

lf_df_t to_load_factor(pax_pdf const& pdf, std::uint16_t capacity);
lf_df_t to_load_factor(pax_cdf const& cdf, std::uint16_t capacity);

void add_additional_groups(
    pax_pdf& pdf, std::vector<additional_group> const& additional_groups);

bool load_factor_possibly_ge(pax_pdf const& pdf, std::uint16_t capacity,
                             float threshold);
bool load_factor_possibly_ge(pax_cdf const& cdf, std::uint16_t capacity,
                             float threshold);
bool load_factor_possibly_ge(lf_df_t const& lf_df, float threshold);

inline float get_pax_le_probability(pax_cdf const& cdf, std::uint16_t limit) {
  return limit < cdf.data_.size() ? cdf.data_[limit] : 0.F;
}

inline float get_pax_gt_probability(pax_cdf const& cdf, std::uint16_t limit) {
  return limit < cdf.data_.size() ? 1.F - cdf.data_[limit] : 0.F;
}

}  // namespace motis::paxmon
