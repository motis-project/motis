#include "motis/paxmon/get_load.h"

namespace motis::paxmon {

std::uint16_t get_base_load(pax_connection_info const& pci) {
  std::uint16_t load = 0;
  for (auto const& si : pci.section_infos_) {
    if (si.valid_ && si.group_->probability_ == 1.0) {
      load += si.group_->passengers_;
    }
  }
  return load;
}

void convolve(pdf_t& pdf, passenger_group const* grp) {
  auto old_pdf = pdf;
  auto const grp_size = grp->passengers_;
  auto const grp_prob = grp->probability_;
  auto const inv_grp_prob = 1.0 - grp_prob;
  for (auto& e : pdf) {
    e.second *= inv_grp_prob;
  }
  for (auto& old : old_pdf) {
    auto const added_size = old.first + grp_size;
    pdf[added_size] += grp_prob * old.second;
  }
}

pdf_t get_load_pdf(pax_connection_info const& pci) {
  auto const base_load = get_base_load(pci);
  auto pdf = pdf_t{};
  pdf[base_load] = 1.0;
  for (auto const& si : pci.section_infos_) {
    if (si.valid_ && si.group_->probability_ != 1.0) {
      convolve(pdf, si.group_);
    }
  }
  return pdf;
}

cdf_t get_cdf(pdf_t const& pdf) {
  auto cdf = cdf_t{};
  auto cumulative_prob = 0.0f;
  for (auto const& e : pdf) {
    cumulative_prob += e.second;
    cdf[e.first] = cumulative_prob;
  }
  return cdf;
}

cdf_t get_load_cdf(pax_connection_info const& pci) {
  return get_cdf(get_load_pdf(pci));
}

}  // namespace motis::paxmon
