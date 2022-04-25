#include "motis/parking/parking_lot.h"

#include <variant>

#include "utl/overloaded.h"

namespace motis::parking {

fee_type parking_lot::get_fee_type() const {
  /*
  return std::visit(utl::overloaded{[](osm_parking_lot_info const& info) {
                                      return info.fee_;
                                    },
                                    [](parkendd_parking_lot_info const& info) {
                                      return fee_type::UNKNOWN;
                                    }},
                    info_);*/
  if (cista::holds_alternative<osm_parking_lot_info>(info_)) {
    return std::get<osm_parking_lot_info>(info_).fee_;
  } else {
    return fee_type::UNKNOWN;
  }
}

}  // namespace motis::parking
