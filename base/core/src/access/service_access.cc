#include "motis/core/access/service_access.h"

#include <string>

#include "boost/algorithm/string/trim.hpp"

#include "motis/core/schedule/schedule.h"
#include "motis/core/access/error.h"

namespace motis {

uint32_t output_train_nr(uint32_t train_nr, uint32_t original_train_nr) {
  return train_nr <= kMaxValidTrainNr ? train_nr : original_train_nr;
}

std::string get_service_name(schedule const& sched,
                             connection_info const* info) {
  auto rule = sched.categories_[info->family_]->output_rule_;
  auto const force_train_nr = (rule & 0b1000U) != 0;  // line_id -> train_nr
  auto const force_provider = (rule & 0b0100U) != 0;  // category -> provider
  auto const base_rule = rule & 0b0011U;

  auto const& line_identifier = info->line_identifier_;
  auto const train_nr =
      output_train_nr(info->train_nr_, info->original_train_nr_);

  std::string print_id;
  if (!line_identifier.empty() &&
      (!force_train_nr || (force_train_nr && train_nr == 0))) {
    print_id = line_identifier;
  } else if (train_nr != 0) {
    print_id = std::to_string(train_nr);
  }

  auto const& category = sched.categories_[info->family_]->name_;

  std::string provider;
  if (info->provider_ != nullptr) {
    provider = info->provider_->short_name_;
  }

  std::string print_cat;
  if (!category.empty() &&
      (!force_provider || (force_provider && provider.empty()))) {
    print_cat = category;
  } else if (!provider.empty()) {
    print_cat = provider;
  }

  switch (base_rule) {
    case 0: {
      auto res = print_cat + " " + print_id;
      boost::algorithm::trim(res);
      return res;
    }
    case 1: return print_cat;
    case 2: return print_id;
    default: return "";
  }
}

}  // namespace motis
