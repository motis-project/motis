#include "motis/path/path.h"

#include <memory>
#include <regex>

#include "geo/polyline.h"
#include "geo/simplify_mask.h"
#include "geo/tile.h"

#include "utl/to_vec.h"
#include "utl/verify.h"

#include "tiles/fixed/convert.h"
#include "tiles/fixed/fixed_geometry.h"
#include "tiles/fixed/io/deserialize.h"
#include "tiles/get_tile.h"
#include "tiles/parse_tile_url.h"

#include "motis/core/common/logging.h"
#include "motis/core/access/realtime_access.h"
#include "motis/core/access/trip_access.h"
#include "motis/core/access/trip_iterator.h"
#include "motis/core/access/trip_section.h"
#include "motis/core/conv/station_conv.h"
#include "motis/core/conv/timestamp_reason_conv.h"
#include "motis/core/conv/transport_conv.h"
#include "motis/core/conv/trip_conv.h"
#include "motis/module/context/get_schedule.h"
#include "motis/module/context/motis_call.h"

#include "motis/path/constants.h"
#include "motis/path/path_database.h"
#include "motis/path/path_index.h"

#include "motis/path/fbs/InternalPathSeqResponse_generated.h"

using namespace flatbuffers;
using namespace motis::module;
using namespace motis::osrm;
using namespace motis::access;
using namespace motis::logging;

namespace motis::path {

struct path::data {
  std::unique_ptr<path_database> db_;
  std::unique_ptr<path_index> index_;

  tiles::render_ctx render_ctx_;
};

path::path() : module("Path", "path"), data_{std::make_unique<path::data>()} {
  param(database_path_, "db", "/path/to/pathdb.mdb");
}

path::~path() = default;

void path::init(registry& r) {
  try {
    data_->db_ = make_path_database(database_path_, true, false);

    data_->render_ctx_ = tiles::make_render_ctx(*data_->db_->db_handle_);

    if (auto buf = data_->db_->try_get(kIndexKey)) {
      data_->index_ = std::make_unique<path_index>(*buf);
    } else {
      LOG(warn) << "pathdb not available: no index!";
    }
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

  // used by: debugger
  r.register_op("/path/by_tile_feature",
                [this](msg_ptr const& m) { return by_tile_feature(m); });

  // used by: debugger
  r.register_op("/path/tiles",
                [this](msg_ptr const& m) { return path_tiles(m); });
}

void path::verify_path_database_available() const {
  if (!data_->db_ || !data_->index_) {
    throw std::system_error(error::database_not_available);
  }
}

msg_ptr path::boxes() const {
  verify_path_database_available();
  auto const boxes = data_->db_->get(kBoxesKey);
  return make_msg(boxes.data(), boxes.size());
};

msg_ptr path::by_station_seq(msg_ptr const& msg) const {
  verify_path_database_available();
  auto req = motis_content(PathByStationSeqRequest, msg);

  return get_response(
      data_->index_->find({utl::to_vec(*req->station_ids(),
                                       [](auto const& s) { return s->str(); }),
                           req->clasz()}),
      req->zoom_level(), req->debug_info());
}

msg_ptr path::by_trip_id(msg_ptr const& msg) const {
  verify_path_database_available();
  auto const& req = motis_content(PathByTripIdRequest, msg);
  auto const& sched = get_schedule();
  auto const& trp = from_fbs(sched, req->trip_id());

  auto const seq = utl::to_vec(access::stops(trp), [&sched](auto const& stop) {
    return stop.get_station(sched).eva_nr_.str();
  });

  auto const s = sections(trp);
  auto const clasz_it =
      std::min_element(begin(s), end(s), [](auto const& lhs, auto const& rhs) {
        return lhs.fcon().clasz_ < rhs.fcon().clasz_;
      });
  utl::verify(clasz_it != end(s), "invalid trip");

  return get_response(data_->index_->find({seq, (*clasz_it).fcon().clasz_}),
                      req->zoom_level(), req->debug_info());
}

Offset<PathSeqResponse> write_response(message_creator& mc,
                                       InternalPathSeqResponse const* resp,
                                       int const zoom_level = -1,
                                       bool const debug_info = true) {
  auto const station_ids =
      utl::to_vec(*resp->station_ids(),
                  [&mc](auto const& e) { return mc.CreateString(e->c_str()); });
  auto const classes = utl::to_vec(*resp->classes());

  auto const segments = utl::to_vec(*resp->segments(), [&](auto const& seg) {
    auto fixed_geo = mpark::get<tiles::fixed_polyline>(
        tiles::deserialize(seg->coords()->str()));
    utl::verify(fixed_geo.size() == 1, "invaild geometry");
    auto& fixed_line = fixed_geo[0];

    if (zoom_level != -1) {
      utl::verify(zoom_level <= 20 && zoom_level > -1, "invalid zoom level");
      geo::apply_simplify_mask(seg->mask()->str(), zoom_level, fixed_line);
    }

    auto serialized_polyline =
        geo::serialize(utl::to_vec(fixed_line, [](auto const& pos) {
          return tiles::fixed_to_latlng(pos);
        }));

    if (zoom_level == -1) {
      return CreateSegment(mc, mc.CreateVector(serialized_polyline),
                           mc.CreateVector(utl::to_vec(*seg->osm_node_ids())));
    } else {
      return CreateSegment(mc, mc.CreateVector(serialized_polyline),
                           mc.CreateVector(std::vector<int64_t>{}));
    }
  });

  auto source_infos = std::vector<Offset<PathSourceInfo>>{};
  if (zoom_level == -1 && debug_info) {
    source_infos = utl::to_vec(*resp->infos(), [&mc](auto const& info) {
      return CreatePathSourceInfo(mc, info->segment_idx(), info->from_idx(),
                                  info->to_idx(),
                                  mc.CreateString(info->type()->c_str()));
    });
  }

  return CreatePathSeqResponse(
      mc, mc.CreateVector(station_ids), mc.CreateVector(classes),
      mc.CreateVector(segments), mc.CreateVector(source_infos));
}

msg_ptr path::by_tile_feature(msg_ptr const& msg) const {
  verify_path_database_available();
  auto const& req = motis_content(PathByTileFeatureRequest, msg);

  message_creator mc;
  std::vector<Offset<PathSeqResponse>> responses;
  for (auto const& [seq, seg] : data_->index_->tile_features_.at(req->ref())) {
    auto buf = data_->db_->get(std::to_string(seq));
    auto original_buf = typed_flatbuffer<motis::path::InternalPathSeqResponse>(
        buf.size(), buf.c_str());

    responses.emplace_back(write_response(mc, original_buf.get()));
  }

  mc.create_and_finish(
      MsgContent_MultiPathSeqResponse,
      CreateMultiPathSeqResponse(mc, mc.CreateVector(responses)).Union());
  return make_msg(mc);
}

msg_ptr path::get_response(std::string const& index, int const zoom_level,
                           bool const debug_info) const {
  verify_path_database_available();
  auto buf = data_->db_->get(index);
  auto original_buf = typed_flatbuffer<motis::path::InternalPathSeqResponse>(
      buf.size(), buf.c_str());
  auto original = original_buf.get();

  message_creator mc;
  mc.create_and_finish(
      MsgContent_PathSeqResponse,
      write_response(mc, original, zoom_level, debug_info).Union());
  return make_msg(mc);
}

msg_ptr path::path_tiles(msg_ptr const& msg) const {
  verify_path_database_available();

  auto tile = tiles::parse_tile_url(msg->get()->destination()->target()->str());
  if (!tile) {
    throw std::system_error(error::invalid_request);
  }

  tiles::null_perf_counter pc;
  auto rendered_tile =
      tiles::get_tile(*data_->db_->db_handle_, *data_->db_->pack_handle_,
                      data_->render_ctx_, *tile, pc);

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
