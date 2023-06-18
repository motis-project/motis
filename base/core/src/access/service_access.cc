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
    auto const& cat_name = sched.categories_[info->family_]->name_;
    auto const& line_id = info->line_identifier_;
    auto const clasz_it = sched.classes_.find(cat_name);
    auto const clasz = clasz_it == end(sched.classes_) ? service_class::OTHER
                                                       : clasz_it->second;
    auto const provider =
        info->provider_ == nullptr ? "" : info->provider_->long_name_;
    if (!line_id.empty() && clasz != service_class::BUS &&
        clasz != service_class::STR &&
        (line_id.view().front() > '9' || line_id.view().front() < '0') &&
        (line_id.view().back() >= '0' && line_id.view().back() <= '9')) {
      // Line ID starts with letter and ends with number, seems to be complete.
      return is(kUseProvider) ? (provider.str() + " " + line_id.str())
                              : line_id.str();
    }

    auto const train_nr =
        output_train_nr(info->train_nr_, info->original_train_nr_);

    auto const second =
        is(kOnlyCategory) ? ""
                          : (train_nr == 0U || ((clasz == service_class::BUS ||
                                                 clasz == service_class::STR ||
                                                 clasz == service_class::S) &&
                                                !line_id.empty())
                                 ? line_id
                                 : fmt::to_string(train_nr));
    auto const omit_s =
        (!second.empty() && second[0] == 'S' && clasz == service_class::S);
    auto const first = is(kOnlyTrainNr) || omit_s
                           ? ""
                           : (is(kUseProvider) ? provider : cat_name);
    return fmt::format("{}{}{}", first, first.empty() ? "" : " ", second);
  }
}

}  // namespace motis
