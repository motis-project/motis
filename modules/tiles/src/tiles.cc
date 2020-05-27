#include "motis/tiles/tiles.h"

#include <cstdlib>

#include "boost/filesystem.hpp"

#include "lmdb/lmdb.hpp"

#include "net/web_server/url_decode.h"

#include "tiles/db/clear_database.h"
#include "tiles/db/feature_inserter_mt.h"
#include "tiles/db/feature_pack.h"
#include "tiles/db/pack_file.h"
#include "tiles/db/prepare_tiles.h"
#include "tiles/db/tile_database.h"
#include "tiles/get_tile.h"
#include "tiles/osm/load_coastlines.h"
#include "tiles/osm/load_osm.h"
#include "tiles/parse_tile_url.h"
#include "tiles/perf_counter.h"

#include "utl/verify.h"

#include "motis/core/common/logging.h"
#include "motis/module/event_collector.h"
#include "motis/module/ini_io.h"

#include "motis/tiles/error.h"

#include "pbf_sdf_fonts_res.h"

namespace c = cista;
namespace mm = motis::module;
namespace ml = motis::logging;
namespace fb = flatbuffers;
namespace fs = boost::filesystem;

namespace motis::tiles {

struct import_state {
  CISTA_COMPARABLE()
  mm::named<c::hash_t, MOTIS_NAME("profile_hash")> profile_hash_;
  mm::named<std::string, MOTIS_NAME("osm_path")> osm_path_;
  mm::named<c::hash_t, MOTIS_NAME("osm_hash")> osm_hash_;
  mm::named<uint64_t, MOTIS_NAME("osm_size")> osm_size_;
  mm::named<bool, MOTIS_NAME("use_coastline")> use_coastline_;
  mm::named<std::string, MOTIS_NAME("coastline_path")> coastline_path_;
  mm::named<c::hash_t, MOTIS_NAME("coastline_hash")> coastline_hash_;
  mm::named<uint64_t, MOTIS_NAME("coastline_size")> coastline_size_;
};

struct tiles::data {
  explicit data(std::string const& path)
      : db_env_{::tiles::make_tile_database(path.c_str())},
        db_handle_{db_env_},
        render_ctx_{::tiles::make_render_ctx(db_handle_)},
        pack_handle_{path.c_str()} {}

  lmdb::env db_env_;
  ::tiles::tile_db_handle db_handle_;
  ::tiles::render_ctx render_ctx_;
  ::tiles::pack_handle pack_handle_;
};

tiles::tiles() : mm::module("Tiles", "tiles") {
  param(profile_path_, "profile", "/path/to/profile.lua");
  param(use_coastline_, "import.use_coastline", "true|false");
}

tiles::~tiles() = default;

void tiles::import(mm::progress_listener& progress_listener,
                   mm::registry& reg) {
  auto const collector = std::make_shared<mm::event_collector>(
      progress_listener, get_data_directory().generic_string(), "tiles", reg,
      [this](std::map<std::string, mm::msg_ptr> const& dependencies) {
        auto const profile_path = fs::path{profile_path_};

        auto profile_hash = c::hash(profile_path.string());
        profile_hash = c::hash_combine(profile_hash, profile_path.size());
        auto const profile_str = utl::read_file(profile_path.string().c_str());
        utl::verify(profile_str.has_value(), "tiles::import cant read profile");
        profile_hash = c::hash_combine(profile_hash, c::hash(*profile_str));

        auto const dir = get_data_directory() / "tiles";
        auto const path = (dir / "tiles.mdb").string();

        using import::OSMEvent;
        auto const osm = motis_content(OSMEvent, dependencies.at("OSM"));
        auto const osm_path = data_path(osm->path()->str());

        auto coastline_path = std::string{};
        auto coastline_hash = c::hash_t{};
        size_t coastline_size{0};
        if (use_coastline_) {
          using import::CoastlineEvent;
          auto const coastline =
              motis_content(CoastlineEvent, dependencies.at("COASTLINE"));
          coastline_path = data_path(coastline->path()->str());
          coastline_hash = coastline->hash();
          coastline_size = coastline->size();
        }

        auto const state = import_state{
            profile_hash,   osm_path,       osm->hash(),    osm->size(),
            use_coastline_, coastline_path, coastline_hash, coastline_size};
        if (mm::read_ini<import_state>(dir / "import.ini") != state) {
          fs::create_directories(dir);

          auto const db_fname = dir / "tiles.mdb";

          ml::clog_import_step("clear database");
          ::tiles::clear_database(path);
          ::tiles::clear_pack_file(path.c_str());

          lmdb::env db_env = ::tiles::make_tile_database(path.c_str());
          ::tiles::tile_db_handle db_handle{db_env};
          ::tiles::pack_handle pack_handle{path.c_str()};

          {
            ::tiles::feature_inserter_mt inserter{
                ::tiles::dbi_handle{db_handle, db_handle.features_dbi_opener()},
                pack_handle};

            if (use_coastline_) {
              ml::clog_import_step("clear database", 0, 20);
              std::clog << '\0' << 'S' << "load coastline" << '\0';
              ::tiles::load_coastlines(db_handle, inserter, coastline_path);
            }

            ml::clog_import_step("load features", 20, 70);
            ::tiles::load_osm(db_handle, inserter, osm_path,
                              profile_path.string());
          }

          ml::clog_import_step("pack features", 70, 90);
          ::tiles::pack_features(db_handle, pack_handle);

          ml::clog_import_step("prepare tiles", 90, 100, 11);
          ::tiles::prepare_tiles(db_handle, pack_handle, 10);
        }

        mm::write_ini(dir / "import.ini", state);
        data_ = std::make_unique<data>(path);
      });
  collector->require("OSM", [](mm::msg_ptr const& msg) {
    return msg->get()->content_type() == MsgContent_OSMEvent;
  });
  if (use_coastline_) {
    collector->require("COASTLINE", [](mm::msg_ptr const& msg) {
      return msg->get()->content_type() == MsgContent_CoastlineEvent;
    });
  }
}

void tiles::init(mm::registry& reg) {
  reg.register_op("/tiles", [&](auto const& msg) {
    auto tile =
        ::tiles::parse_tile_url(msg->get()->destination()->target()->str());
    if (!tile) {
      throw std::system_error(error::invalid_request);
    }

    ::tiles::null_perf_counter pc;
    auto rendered_tile = ::tiles::get_tile(
        data_->db_handle_, data_->pack_handle_, data_->render_ctx_, *tile, pc);

    mm::message_creator mc;
    std::vector<fb::Offset<HTTPHeader>> headers;
    fb::Offset<fb::String> payload;
    if (rendered_tile) {
      headers.emplace_back(CreateHTTPHeader(
          mc, mc.CreateString("Content-Type"),
          mc.CreateString("application/vnd.mapbox-vector-tile")));
      headers.emplace_back(CreateHTTPHeader(
          mc, mc.CreateString("Content-Encoding"), mc.CreateString("deflate")));
      payload = mc.CreateString(rendered_tile->data(), rendered_tile->size());
    } else {
      payload = mc.CreateString("");
    }

    mc.create_and_finish(
        MsgContent_HTTPResponse,
        CreateHTTPResponse(mc, HTTPStatus_OK, mc.CreateVector(headers), payload)
            .Union());

    return make_msg(mc);
  });

  reg.register_op("/tiles/glyphs", [&](auto const& msg) {
    std::string decoded;
    net::url_decode(msg->get()->destination()->target()->str(), decoded);
    auto const mem = pbf_sdf_fonts_res::get_resource(decoded.substr(14));

    mm::message_creator mc;
    mc.create_and_finish(
        MsgContent_HTTPResponse,
        CreateHTTPResponse(
            mc, HTTPStatus_OK,
            mc.CreateVector(std::vector<fb::Offset<HTTPHeader>>{}),
            mc.CreateString(reinterpret_cast<char const*>(mem.ptr_), mem.size_))
            .Union());
    return make_msg(mc);
  });
}

bool tiles::import_successful() const { return data_ != nullptr; }

}  // namespace motis::tiles
