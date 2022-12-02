#include "motis/paxmon/get_load.h"
#include "motis/paxmon/get_load_internal.h"

#include <cassert>
#include <algorithm>
#include <numeric>

#include "utl/enumerate.h"
#include "utl/verify.h"

namespace motis::paxmon {

std::uint16_t get_expected_load(pci_container const& pcis,
                                pci_index const idx) {
  return pcis.expected_load_[idx];
}

std::uint16_t get_expected_load(universe const& uv, pci_index const idx) {
  return uv.pax_connection_info_.expected_load_[idx];
}

std::uint16_t get_pax_quantile(pax_cdf const& cdf, float const q) {
  for (auto const& [pax, prob] : utl::enumerate(cdf.data_)) {
    if (prob >= q) {
      return pax;
    }
  }
  throw utl::fail("get_pax_quantile: invalid cdf");
}

std::uint16_t get_median_load(pax_cdf const& cdf) {
  return get_pax_quantile(cdf, 0.5F);
}

pax_stats get_pax_stats(pax_cdf const& cdf) {
  auto stats = pax_stats{};
  if (!cdf.data_.empty()) {
    auto min_found = false;
    auto q5_found = false;
    auto q50_found = false;
    auto q95_found = false;
    for (auto const& [pax, prob] : utl::enumerate(cdf.data_)) {
      if (!min_found && prob != 0.F) {
        stats.limits_.min_ = pax;
        min_found = true;
      }
      if (!q5_found && prob >= 0.05F) {
        stats.q5_ = pax;
        q5_found = true;
      }
      if (!q50_found && prob >= 0.5F) {
        stats.q50_ = pax;
        q50_found = true;
      }
      if (!q95_found && prob >= 0.95F) {
        stats.q95_ = pax;
        q95_found = true;
      }
    }
    stats.limits_.max_ = cdf.data_.size();
  }
  return stats;
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
    std::vector<additional_group> const& additional_groups) {
  return std::accumulate(
      begin(additional_groups), end(additional_groups), 0ULL,
      [](auto const sum, auto const& ag) { return sum + ag.passengers_; });
}

void add_additional_groups_base(
    pax_pdf& pdf, std::vector<additional_group> const& additional_groups) {
  assert(!additional_groups.empty());
  auto const max_new_pax = get_max_new_pax(additional_groups);
  pdf.data_.resize(pdf.data_.size() + max_new_pax);
  for (auto const& ag : additional_groups) {
    convolve_base(pdf, ag.passengers_, ag.probability_);
  }
}

#ifdef MOTIS_AVX2
void add_additional_groups_avx(
    pax_pdf& pdf, std::vector<additional_group> const& additional_groups) {
  assert(!additional_groups.empty());
  auto const max_new_pax = get_max_new_pax(additional_groups);
  auto const pdf_size = pdf.data_.size() + max_new_pax;
  pdf.data_.resize(round_up<8>(pdf_size));
  auto buf = std::vector<float>(pdf.data_.size() + 8);
  auto limits = pax_limits{
      std::min_element(begin(additional_groups), end(additional_groups),
                       [](auto const& ag1, auto const& ag2) {
                         return ag1.passengers_ < ag2.passengers_;
                       })
          ->passengers_,
      0 /* not used */};
  for (auto const& ag : additional_groups) {
    convolve_avx(pdf, ag.passengers_, ag.probability_, limits, buf);
  }
  pdf.data_.resize(pdf_size);
}
#endif

void add_additional_groups(
    pax_pdf& pdf, std::vector<additional_group> const& additional_groups) {
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
    // TODO(pablo): maybe switch back to max
    // return cdf.data_.back() != 0.0;
    if (cdf.data_[pax_threshold] <= 0.95F) {
      return true;
    } else {
      return pax_threshold > 0 ? cdf.data_[pax_threshold - 1] < 0.95F : false;
    }
  }
}

bool load_factor_possibly_ge(lf_df_t const& lf_df, float threshold) {
  return lf_df.lower_bound(threshold) != end(lf_df);
}

}  // namespace motis::paxmon
