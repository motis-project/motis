#pragma once

#include "motis/core/common/logging.h"

#include "motis/module/context/motis_call.h"
#include "motis/module/message.h"

#include <map>
#include <string>

namespace motis::intermodal {

struct ppr_profiles {
  static constexpr auto const DEFAULT_PPR_WALKING_SPEED = 1.4;

  void update() {
    using namespace motis::module;
    using namespace motis::ppr;
    try {
      auto const msg = motis_call(make_no_msg("/ppr/profiles"))->val();
      auto const profiles = motis_content(FootRoutingProfilesResponse, msg);
      walking_speed_.clear();
      for (auto const& pi : *profiles->profiles()) {
        walking_speed_[pi->name()->str()] = pi->walking_speed();
      }
    } catch (std::system_error const& e) {
      LOG(logging::warn) << "intermodal: ppr profiles not loaded: " << e.what();
    }
  }

  inline double get_walking_speed(std::string const& profile) const {
    auto const it = walking_speed_.find(profile);
    return it != end(walking_speed_) ? it->second : DEFAULT_PPR_WALKING_SPEED;
  }

private:
  std::map<std::string, double> walking_speed_;
};

}  // namespace motis::intermodal
