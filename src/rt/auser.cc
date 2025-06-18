#include "motis/rt/auser.h"

#include "boost/json.hpp"
#include "pugixml.hpp"

namespace motis {

auser::auser(const nigiri::timetable& tt, nigiri::source_idx_t s)
    : upd_{tt, s} {}

std::string auser::fetch_url(std::string_view base_url) {
  return fmt::format("{}/api/v1/auser/fetch?since={}", base_url, last_update_);
}

nigiri::rt::vdv::statistics auser::consume_update(std::string auser_update,
                                                  nigiri::rt_timetable& rtt) {
  auto j = boost::json::parse(auser_update).as_object();
  last_update_ = j["id"].as_string();
  auto vdvaus = pugi::xml_document{};
  vdvaus.load_string(j["update"].as_string().c_str());
  upd_.update(rtt, vdvaus);
  return upd_.get_stats();
}

}  // namespace motis
