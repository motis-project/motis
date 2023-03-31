#include "motis/valhalla/valhalla.h"

#include "baldr/rapidjson_utils.h"
#include "config.h"
#include "filesystem.h"
#include "midgard/logging.h"
#include "midgard/util.h"
#include "mjolnir/util.h"

using namespace motis::module;

namespace motis::valhalla {

valhalla::valhalla() : module("Valhalla Street Router", "valhalla") {}

valhalla::~valhalla() noexcept = default;

void valhalla::init(motis::module::registry& reg) {
  reg.register_op("/valhalla",
                  [&](msg_ptr const&) { return make_success_msg(); }, {});
  //  ::valhalla::mjolnir::build_tile_set();
}

}  // namespace motis::valhalla
