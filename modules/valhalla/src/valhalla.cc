#include "motis/valhalla/valhalla.h"

#include <filesystem>

#include "cista/reflection/comparable.h"

#include "baldr/rapidjson_utils.h"
#include "config.h"
#include "filesystem.h"
#include "midgard/logging.h"
#include "midgard/util.h"
#include "mjolnir/util.h"

#include "motis/module/event_collector.h"
#include "motis/module/ini_io.h"

namespace mm = motis::module;
namespace fs = std::filesystem;
namespace v = ::valhalla;

namespace motis::valhalla {

std::string get_config(std::string const& tile_dir) {
  return fmt::format(R"({
  "mjolnir": {
    "tile_dir": "{}",
    "data_processing": {
      "use_admin_db": false
    }
  }
})",
                     tile_dir);
}

struct import_state {
  CISTA_COMPARABLE()
  mm::named<std::string, MOTIS_NAME("path")> path_;
  mm::named<cista::hash_t, MOTIS_NAME("hash")> hash_;
  mm::named<size_t, MOTIS_NAME("size")> size_;
};

valhalla::valhalla() : module("Valhalla Street Router", "valhalla") {}

valhalla::~valhalla() noexcept = default;

void valhalla::init(mm::registry& reg) {
  reg.register_op("/valhalla",
                  [&](mm::msg_ptr const& msg) { return route(msg); }, {});
}

mm::msg_ptr valhalla::route(mm::msg_ptr const& msg) {
  return mm::make_success_msg();
}

void valhalla::import(mm::import_dispatcher& reg) {
  std::make_shared<mm::event_collector>(
      get_data_directory().generic_string(), "valhalla", reg,
      [this](mm::event_collector::dependencies_map_t const& dependencies,
             mm::event_collector::publish_fn_t const& publish) {
        using import::OSMEvent;

        auto const osm = motis_content(OSMEvent, dependencies.at("OSM"));
        auto const state = import_state{data_path(osm->path()->str()),
                                        osm->hash(), osm->size()};

        auto const osm_stem = fs::path{fs::path{osm->path()->str()}.stem()}
                                  .stem()
                                  .generic_string();

        auto const dir = get_data_directory() / "valhalla";
        fs::create_directories(dir);

        if (mm::read_ini<import_state>(dir / "import.ini") != state) {
          auto const config = get_config(dir);

          std::stringstream ss;
          ss << config;

          boost::property_tree::ptree pt;
          rapidjson::read_json(ss, pt);

          v::mjolnir::build_tile_set(pt, {osm->path()->str()});
        }
      })
      ->require("OSM", [](mm::msg_ptr const& msg) {
        return msg->get()->content_type() == MsgContent_OSMEvent;
      });
}

}  // namespace motis::valhalla
