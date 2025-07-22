#include "motis/rt/auser.h"

#include "pugixml.hpp"

#include "nigiri/common/parse_time.h"

#include "motis/http_req.h"

namespace n = nigiri;

namespace motis {

auser::auser(nigiri::timetable const& tt,
             n::source_idx_t const s,
             n::rt::vdv_aus::updater::xml_format const format)
    : upd_{tt, s, format} {}

std::string auser::fetch_url(std::string_view base_url) {
  return upd_.get_format() == n::rt::vdv_aus::updater::xml_format::kVdv
             ? fmt::format("{}/auser/fetch?since={}&body_limit={}", base_url,
                           update_state_, kBodySizeLimit)
             : std::string{base_url};
}

n::rt::vdv_aus::statistics auser::consume_update(
    std::string const& auser_update, n::rt_timetable& rtt) {
  auto vdvaus = pugi::xml_document{};
  vdvaus.load_string(auser_update.c_str());
  auto stats = upd_.update(rtt, vdvaus);

  try {
    auto const prev_update = update_state_;
    update_state_ =
        upd_.get_format() == n::rt::vdv_aus::updater::xml_format::kVdv
            ? vdvaus.select_node("//AUSNachricht")
                  .node()
                  .attribute("auser_id")
                  .as_llong(0ULL)
            : n::parse_time_no_tz(vdvaus.select_node("//RecordedAtTime")
                                      .node()
                                      .text()
                                      .as_string())
                  .time_since_epoch()
                  .count();
    fmt::println("[auser] {} --> {}", prev_update, update_state_);
  } catch (...) {
  }

  return stats;
}

}  // namespace motis
