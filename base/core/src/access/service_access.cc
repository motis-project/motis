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
  constexpr auto const kOnlyCategory = std::uint8_t{0b0001};
  constexpr auto const kOnlyTrainNr = std::uint8_t{0b0010};
  constexpr auto const kNoOutput = std::uint8_t{0b0011};
  constexpr auto const kUseProvider = std::uint8_t{0b1000};

  auto const rule = sched.categories_[info->family_]->output_rule_;
  auto const is = [&](auto const flag) { return (rule & flag) == flag; };

  if (is(kNoOutput)) {
    return "";
  } else {
    auto const train_nr =
        output_train_nr(info->train_nr_, info->original_train_nr_);
    auto const line_id = info->line_identifier_;
    auto const& cat = *sched.categories_[info->family_];
    std::string provider;
    if (info->provider_ != nullptr) {
      provider = info->provider_->short_name_;
    }
    auto const first =
        is(kOnlyTrainNr) ? "" : (is(kUseProvider) ? provider : cat.name_);
    auto const second =
        is(kOnlyCategory)
            ? ""
            : (train_nr == 0U ? line_id : fmt::to_string(train_nr));
    return fmt::format("{}{}{}", first, first.empty() ? "" : " ", second);
  }
}

}  // namespace motis
