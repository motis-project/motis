#include "motis/paxmon/get_load.h"

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

inline void convolve(pax_pdf& pdf, std::uint16_t const grp_size,
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

inline void convolve(pax_pdf& pdf, passenger_group const* grp) {
  convolve(pdf, grp->passengers_, grp->probability_);
}

pax_pdf get_load_pdf(pax_connection_info const& pci) {
  auto const limits = get_pax_limits(pci);
  auto pdf = pax_pdf{};
  pdf.data_.resize(limits.max_ + 1);
  pdf.data_[limits.min_] = 1.0F;
  for (auto const& si : pci.section_infos_) {
    if (si.group_->probability_ != 1.0F) {
      convolve(pdf, si.group_);
    }
  }
  return pdf;
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
  convolve(pdf, passengers, probability);
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
