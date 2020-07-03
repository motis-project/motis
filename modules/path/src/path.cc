#include "motis/path/path.h"

#include <memory>

#include "boost/filesystem.hpp"

#include "geo/polyline.h"
#include "geo/tile.h"

#include "utl/to_vec.h"
#include "utl/verify.h"

#include "tiles/fixed/io/deserialize.h"
#include "tiles/get_tile.h"
#include "tiles/parse_tile_url.h"

#include "motis/core/common/logging.h"
#include "motis/core/access/trip_iterator.h"
#include "motis/core/conv/trip_conv.h"
#include "motis/module/context/get_schedule.h"
#include "motis/module/context/motis_call.h"
#include "motis/module/event_collector.h"
#include "motis/module/ini_io.h"

#include "motis/path/definitions.h"
#include "motis/path/error.h"
#include "motis/path/path_data.h"
#include "motis/path/path_database_query.h"

#include "motis/path/prepare/prepare.h"

using namespace flatbuffers;
using namespace motis::module;
using namespace motis::logging;

namespace motis::path {

struct import_state {
  CISTA_COMPARABLE()
  named<std::string, MOTIS_NAME("osm_path")> osm_path_;
  named<cista::hash_t, MOTIS_NAME("osm_hash")> osm_hash_;
  named<size_t, MOTIS_NAME("osm_size")> osm_size_;
  named<cista::hash_t, MOTIS_NAME("schedule_hash")> schedule_hash_;
};

path::path() : module("Path", "path") {}

path::~path() = default;

void path::import(registry& reg) {
  std::make_shared<event_collector>(
      get_data_directory().generic_string(), "path", reg,
      [this](std::map<std::string, msg_ptr> const& dependencies) {
        using import::ScheduleEvent;
        using import::OSMEvent;
        using import::OSRMEvent;
        auto const schedule =
            motis_content(ScheduleEvent, dependencies.at("SCHEDULE"));
        auto const osm = motis_content(OSMEvent, dependencies.at("OSM"));
        auto const osrm = motis_content(OSRMEvent, dependencies.at("OSRM"));

        auto const dir = get_data_directory() / "path";
        boost::filesystem::create_directories(dir);

        auto const state =
            import_state{data_path(osm->path()->str()), osm->hash(),
                         osm->size(), schedule->hash()};
        if (read_ini<import_state>(dir / "import.ini") == state) {
          import_successful_ = true;
          return;
        }

        prepare_settings opt;
        opt.schedule_ = schedule->raw_file()->str();
        opt.osm_ = osm->path()->str();
        opt.osrm_ = osrm->path()->str();
        opt.out_ = (dir / "pathdb.mdb").generic_string();
        opt.tmp_ = dir.generic_string();
        prepare(opt);

        write_ini(dir / "import.ini", state);
        import_successful_ = true;
      })
      ->require("OSM",
                [](msg_ptr const& msg) {
                  return msg->get()->content_type() == MsgContent_OSMEvent;
                })
      ->require("SCHEDULE",
                [](msg_ptr const& msg) {
                  return msg->get()->content_type() == MsgContent_ScheduleEvent;
                })
      ->require("OSRM", [](msg_ptr const& msg) {
        using import::OSRMEvent;
        return msg->get()->content_type() == MsgContent_OSRMEvent &&
               motis_content(OSRMEvent, msg)->profile()->str() == "bus";
      });
}

void path::init(registry& r) {
  try {
    auto data = path_data{};
    data.db_ = make_path_database(
        (get_data_directory() / "path" / "pathdb.mdb").generic_string(), true,
        false);

    data.render_ctx_ = tiles::make_render_ctx(*data.db_->db_handle_);

    if (auto buf = data.db_->try_get(kIndexKey); buf.has_value()) {
      data.index_ = std::make_unique<path_index>(*buf);
    } else {
      LOG(warn) << "pathdb not available: no index!";
    }

    add_shared_data(PATH_DATA_KEY, std::move(data));
  } catch (std::system_error const&) {
    LOG(warn) << "pathdb not available: no database!";
  }

  // used by: railviz
  r.register_op("/path/boxes", [this](msg_ptr const&) { return boxes(); });

  // used by: railviz, sim, legacydebugger
  r.register_op("/path/by_trip_id",
                [this](msg_ptr const& m) { return by_trip_id(m); });

  // used by: sim, legacydebugger
  r.register_op("/path/by_station_seq",
                [this](msg_ptr const& m) { return by_station_seq(m); });

  // used by: railviz
  r.register_op("/path/by_trip_id_batch",
                [this](msg_ptr const& m) { return by_trip_id_batch(m); });

  // used by: debugger
  r.register_op("/path/by_tile_feature",
                [this](msg_ptr const& m) { return by_tile_feature(m); });

  // used by: debugger
  r.register_op("/path/tiles",
                [this](msg_ptr const& m) { return path_tiles(m); });
}

msg_ptr path::boxes() const {
  auto const& data = get_shared_data<path_data>(PATH_DATA_KEY);
  auto const boxes = data.db_->get(kBoxesKey);
  return make_msg(boxes.data(), boxes.size());
};

msg_ptr path::by_station_seq(msg_ptr const& msg) const {
  auto const& data = get_shared_data<path_data>(PATH_DATA_KEY);
  auto req = motis_content(PathByStationSeqRequest, msg);

  return data.get_response(
      data.index_->find(
          {utl::to_vec(*req->station_ids(),
                       [](auto const& s) { return s->str(); }),
           service_class{static_cast<service_class>(req->clasz())}}),
      req->zoom_level());
}

msg_ptr path::by_trip_id(msg_ptr const& msg) const {
  auto const& data = get_shared_data<path_data>(PATH_DATA_KEY);
  auto const& req = motis_content(PathByTripIdRequest, msg);
  auto const& sched = get_schedule();
  return data.get_response(
      data.trip_to_index(sched, from_fbs(sched, req->trip_id())),
      req->zoom_level());
}

msg_ptr path::by_trip_id_batch(msg_ptr const& msg) const {
  auto const& data = get_shared_data<path_data>(PATH_DATA_KEY);
  auto const& req = motis_content(PathByTripIdBatchRequest, msg);
  auto const& sched = get_schedule();

  path_database_query q{req->zoom_level()};

  for (auto const* trp_segment : *req->trip_segments()) {
    auto const* trp = from_fbs(sched, trp_segment->trip_id());
    auto segments = utl::to_vec(*trp_segment->segments(),
                                [](auto s) -> size_t { return s; });

    try {
      auto index = data.trip_to_index(sched, trp);
      q.add_sequence(index, std::move(segments));
    } catch (std::system_error const&) {
      std::vector<geo::polyline> extra;
      size_t i = 0;
      for (auto const& s : access::sections(trp)) {
        if (segments.empty() ||
            std::find(begin(segments), end(segments), i) != end(segments)) {
          auto const& from = s.from_station(sched);
          auto const& to = s.to_station(sched);
          extra.emplace_back(geo::polyline{geo::latlng{from.lat(), from.lng()},
                                           geo::latlng{to.lat(), to.lng()}});
        }
        ++i;
      }
      q.add_extra(extra);
    }
  }

  q.execute(*data.db_);

  message_creator mc;
  mc.create_and_finish(MsgContent_PathByTripIdBatchResponse,
                       q.write_batch(mc).Union());
  return make_msg(mc);
}

msg_ptr path::by_tile_feature(msg_ptr const& msg) const {
  auto const& data = get_shared_data<path_data>(PATH_DATA_KEY);
  auto const& req = motis_content(PathByTileFeatureRequest, msg);

  message_creator mc;
  std::vector<Offset<PathSeqResponse>> responses;
  for (auto const& [seq, seg] : data.index_->tile_features_.at(req->ref())) {
    // TODO(sebastian) update this for the batch query
    responses.emplace_back(data.reconstruct_sequence(mc, seq));
  }

  mc.create_and_finish(
      MsgContent_MultiPathSeqResponse,
      CreateMultiPathSeqResponse(mc, mc.CreateVector(responses)).Union());
  return make_msg(mc);
}

msg_ptr path::path_tiles(msg_ptr const& msg) const {
  auto const& data = get_shared_data<path_data>(PATH_DATA_KEY);
  auto tile = tiles::parse_tile_url(msg->get()->destination()->target()->str());
  utl::verify_ex(tile.has_value(), std::system_error{error::invalid_request});

  tiles::null_perf_counter pc;
  auto rendered_tile =
      tiles::get_tile(*data.db_->db_handle_, *data.db_->pack_handle_,
                      data.render_ctx_, *tile, pc);

  message_creator mc;
  std::vector<Offset<HTTPHeader>> headers;
  Offset<String> payload;
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
}

}  // namespace motis::path
