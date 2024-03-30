#include "motis/adr/adr.h"

#include <filesystem>
#include <fstream>
#include <istream>
#include <regex>
#include <sstream>

#include "cista/reflection/comparable.h"

#include "utl/to_vec.h"

#include "adr/adr.h"

#include "motis/core/common/logging.h"
#include "motis/module/event_collector.h"
#include "motis/module/ini_io.h"

namespace mm = motis::module;
namespace a = adr;

namespace motis::adr {

struct import_state {
  CISTA_COMPARABLE()
  mm::named<std::string, MOTIS_NAME("path")> path_;
  mm::named<cista::hash_t, MOTIS_NAME("hash")> hash_;
  mm::named<size_t, MOTIS_NAME("size")> size_;
};

struct adr::impl {
};

adr::adr() : module("Address Typeahead", "adr") {}

adr::~adr() = default;

void adr::import(motis::module::import_dispatcher& reg) {
  std::make_shared<mm::event_collector>(
      get_data_directory().generic_string(), "adr", reg,
      [this](mm::event_collector::dependencies_map_t const& dependencies,
             mm::event_collector::publish_fn_t const&) {
        using import::OSMEvent;

        auto const dir = get_data_directory() / "adr";
        auto const osm = motis_content(OSMEvent, dependencies.at("OSM"));
        auto const state = import_state{data_path(osm->path()->str()),
                                        osm->hash(), osm->size()};

        if (mm::read_ini<import_state>(dir / "import.ini") != state) {
          mm::write_ini(dir / "import.ini", state);
        }

        import_successful_ = true;
      })
      ->require("OSM", [](mm::msg_ptr const& msg) {
        return msg->get()->content_type() == MsgContent_OSMEvent;
      });
}

void adr::init(mm::registry& reg) {
  reg.register_op(
      "/adr",
      [this](mm::msg_ptr const& msg) { (void) msg; return mm::make_success_msg(); }, {});
}

}  // namespace motis::adr
