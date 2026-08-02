#pragma once

#include "utl/helpers/algorithm.h"
#include "utl/to_vec.h"

#include "nigiri/rt/service_alert.h"
#include "nigiri/types.h"

#include "motis-api/motis-api.h"

namespace motis {

inline api::Alert to_alert(
    nigiri::alerts const& al,
    nigiri::alert_idx_t const x,
    std::optional<std::vector<std::string>> const& lang) {
  auto const convert_to_str = [](std::string_view s) {
    return std::optional{std::string{s}};
  };
  auto const get_translation =
      [&](auto const& translations) -> std::optional<std::string> {
    if (translations.empty()) {
      return std::nullopt;
    } else if (!lang.has_value()) {
      return al.strings_.try_get(translations.front().text_)
          .and_then(convert_to_str);
    } else {
      for (auto const& req_lang : *lang) {
        auto const it = utl::find_if(
            translations, [&](nigiri::alert_translation const translation) {
              auto const translation_lang =
                  al.strings_.try_get(translation.language_);
              return translation_lang.has_value() &&
                     translation_lang->starts_with(req_lang);
            });
        if (it == end(translations)) {
          continue;
        }
        return al.strings_.try_get(it->text_).and_then(convert_to_str);
      }
      return al.strings_.try_get(translations.front().text_)
          .and_then(convert_to_str);
    }
  };
  auto const to_time_range =
      [](nigiri::interval<nigiri::unixtime_t> const x) -> api::TimeRange {
    return {x.from_, x.to_};
  };
  return {
      .communicationPeriod_ =
          al.communication_period_[x].empty()
              ? std::nullopt
              : std::optional{utl::to_vec(al.communication_period_[x],
                                          to_time_range)},
      .impactPeriod_ =
          al.impact_period_[x].empty()
              ? std::nullopt
              : std::optional{utl::to_vec(al.impact_period_[x], to_time_range)},
      .cause_ = api::AlertCauseEnum{static_cast<int>(al.cause_[x])},
      .causeDetail_ = get_translation(al.cause_detail_[x]),
      .effect_ = api::AlertEffectEnum{static_cast<int>(al.effect_[x])},
      .effectDetail_ = get_translation(al.effect_detail_[x]),
      .url_ = get_translation(al.url_[x]),
      .headerText_ = get_translation(al.header_text_[x]).value_or(""),
      .descriptionText_ = get_translation(al.description_text_[x]).value_or(""),
      .ttsHeaderText_ = get_translation(al.tts_header_text_[x]),
      .ttsDescriptionText_ = get_translation(al.tts_description_text_[x]),
      .imageUrl_ = al.image_[x].empty()
                       ? std::nullopt
                       : al.strings_.try_get(al.image_[x].front().url_)
                             .and_then(convert_to_str),
      .imageMediaType_ =
          al.image_[x].empty()
              ? std::nullopt
              : al.strings_.try_get(al.image_[x].front().media_type_)
                    .and_then(convert_to_str),
      .imageAlternativeText_ = get_translation(al.image_alternative_text_[x])};
}

}  // namespace motis
