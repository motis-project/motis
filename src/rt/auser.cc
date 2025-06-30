#include "motis/rt/auser.h"

#include "pugixml.hpp"

#include "motis/http_req.h"

namespace motis {

auser::auser(const nigiri::timetable& tt, nigiri::source_idx_t s)
    : upd_{tt, s} {}

std::string auser::fetch_url(std::string_view base_url) {
  return fmt::format("{}/auser/fetch?since={}&body_limit={}", base_url,
                     update_state_, kBodySizeLimit);
}

nigiri::rt::vdv_aus::statistics auser::consume_update(
    std::string const& auser_update, nigiri::rt_timetable& rtt) {
  auto vdvaus = pugi::xml_document{};
  vdvaus.load_string(auser_update.c_str());
  upd_.update(rtt, vdvaus);
  auto const prev_update = update_state_;
  update_state_ = std::stol(vdvaus.select_node("//AUSNachricht")
                                .node()
                                .attribute("auser_id")
                                .as_string());

  fmt::println("[auser] {} --> {}", prev_update, update_state_);

  return upd_.get_stats();
}

}  // namespace motis
