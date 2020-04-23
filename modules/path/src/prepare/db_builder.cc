#include "motis/path/prepare/db_builder.h"

#include <mutex>

#include "geo/box.h"
#include "geo/polyline.h"
#include "geo/simplify_mask.h"
#include "geo/webmercator.h"

#include "utl/get_or_create.h"
#include "utl/parser/arg_parser.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "tiles/db/feature_inserter_mt.h"
#include "tiles/db/layer_names.h"
#include "tiles/db/prepare_tiles.h"
#include "tiles/db/tile_database.h"
#include "tiles/feature/feature.h"
#include "tiles/fixed/convert.h"
#include "tiles/fixed/fixed_geometry.h"
#include "tiles/fixed/io/serialize.h"

#include "motis/hash_map.h"

#include "motis/core/common/logging.h"
#include "motis/module/message.h"

#include "motis/path/constants.h"
#include "motis/path/path_database.h"

#include "motis/path/fbs/InternalPathSeqResponse_generated.h"
#include "motis/path/fbs/PathIndex_generated.h"
#include "motis/protocol/PathSeqResponse_generated.h"

using namespace flatbuffers;
using namespace motis::module;

namespace motis::path {

using seq_info =
    std::tuple<std::vector<std::string>, std::vector<uint32_t>, int>;

template <typename Classes>
uint64_t cls_to_min_zoom_level(Classes const& c) {
  auto it = std::min_element(begin(c), end(c));
  utl::verify(it != end(c), "classes container empty");

  if (*it < 3) {
    return 4UL;
  } else if (*it < 6) {
    return 6UL;
  } else if (*it < 7) {
    return 8UL;
  } else {
    return 9UL;
  }
}

template <typename Classes>
uint64_t cls_to_bits(Classes const& c) {
  uint64_t class_bits = 0;
  for (auto const& cls : c) {
    class_bits |= 1UL << static_cast<size_t>(cls);
  }
  return class_bits;
}

struct db_builder::impl {
  explicit impl(std::string const& fname)
      : db_(make_path_database(fname, false, true)),
        feature_inserter_{std::make_unique<tiles::feature_inserter_mt>(
            tiles::dbi_handle{*db_->db_handle_,
                              db_->db_handle_->features_dbi_opener()},
            *db_->pack_handle_)} {
    tiles::layer_names_builder layer_names;
    station_layer_id_ = layer_names.get_layer_idx("station");
    path_layer_id_ = layer_names.get_layer_idx("path");

    auto txn = db_->db_handle_->make_txn();
    layer_names.store(*db_->db_handle_, txn);
    txn.commit();
  }

  void store_stations(std::vector<station> const& stations) const {
    for (auto const& s : stations) {
      if (std::none_of(begin(s.categories_), end(s.categories_),
                       [](auto cls) { return cls < 9; })) {
        continue;
      }

      tiles::feature f;
      auto cstr = utl::cstr(s.id_.c_str());
      utl::parse_arg(cstr, f.id_, 0);
      f.layer_ = station_layer_id_;
      f.zoom_levels_ = {cls_to_min_zoom_level(s.categories_),
                        tiles::kMaxZoomLevel};

      f.meta_.emplace_back("name", tiles::encode_string(s.name_));
      f.meta_.emplace_back(
          "classes",
          tiles::encode_string(std::to_string(cls_to_bits(s.categories_))));

      f.geometry_ = tiles::fixed_point{
          {tiles::latlng_to_fixed({s.pos_.lat_, s.pos_.lng_})}};

      feature_inserter_->insert(f);
    }
  }

  using internal_response =
      typed_flatbuffer<motis::path::InternalPathSeqResponse>;

  static internal_response serialize_seq(
      resolved_station_seq const& seq,
      std::vector<processed_segment> const& processed) {
    message_creator mc;
    auto const fbs_stations = utl::to_vec(
        seq.station_ids_, [&](auto const& id) { return mc.CreateString(id); });

    auto fbs_segments = utl::to_vec(processed, [&](auto const& proc) {
      tiles::fixed_polyline polyline;
      polyline.emplace_back();
      polyline.back().reserve(proc.polyline_.size());
      for (auto const& pos : proc.polyline_) {
        polyline.back().emplace_back(tiles::latlng_to_fixed(pos));
      }

      return CreateInternalSegment(
          mc, mc.CreateString(tiles::serialize(polyline)),
          mc.CreateString(geo::serialize_simplify_mask(proc.mask_)),
          mc.CreateVector(proc.osm_ids_));
    });

    std::vector<Offset<InternalPathSourceInfo>> fbs_info;
    for (auto const& info : seq.sequence_infos_) {
      fbs_info.push_back(CreateInternalPathSourceInfo(
          mc, info.idx_, info.from_, info.to_, info.between_stations_,
          static_cast<std::underlying_type_t<source_spec::category>>(
              info.source_spec_.category_),
          static_cast<std::underlying_type_t<source_spec::router>>(
              info.source_spec_.router_)));
    }

    mc.Finish(CreateInternalPathSeqResponse(
        mc, mc.CreateVector(fbs_stations), mc.CreateVector(seq.classes_),
        mc.CreateVector(fbs_segments), mc.CreateVector(fbs_info)));

    return internal_response{std::move(mc)};
  }

  void add_seq(size_t seq_idx, resolved_station_seq const& seq,
               std::vector<processed_segment> const& processed) {
    auto const& boxes = utl::to_vec(
        seq.paths_, [](auto const& path) { return geo::box{path.polyline_}; });

    auto internal_resp = serialize_seq(seq, processed);

    std::lock_guard<std::mutex> lock(m_);
    update_boxes(seq.station_ids_, boxes);

    db_put(std::to_string(seq_idx), internal_resp.to_string());
    seq_infos_.emplace_back(seq.station_ids_, seq.classes_, seq_idx);
  }

  void update_boxes(std::vector<std::string> const& station_ids,
                    std::vector<geo::box> const& boxes) {
    for (auto i = 0UL; i < boxes.size(); ++i) {
      auto key = (station_ids[i] < station_ids[i + 1])
                     ? std::make_pair(station_ids[i], station_ids[i + 1])
                     : std::make_pair(station_ids[i + 1], station_ids[i]);

      utl::get_or_create(boxes_, key, [] {
        return geo::box{};
      }).extend(boxes[i]);
    }
  }

  void add_tile_feature(geo::polyline const& line,
                        std::vector<seq_seg> const& seq_segs,
                        std::vector<uint32_t> const& classes, bool is_stub) {
    if (!classes.empty() && std::none_of(begin(classes), end(classes),
                                         [](auto c) { return c < 9; })) {
      return;
    }

    tiles::feature f;
    f.layer_ = path_layer_id_;
    f.zoom_levels_ = {cls_to_min_zoom_level(classes), tiles::kMaxZoomLevel};

    f.meta_.emplace_back(
        "classes", tiles::encode_string(std::to_string(cls_to_bits(classes))));
    f.meta_.emplace_back("min_class", tiles::encode_integer(*std::min_element(
                                          begin(classes), end(classes))));
    f.meta_.emplace_back("stub", tiles::encode_bool(is_stub));

    tiles::fixed_polyline polyline;
    polyline.emplace_back();
    polyline.back().reserve(line.size());
    for (auto const& pos : line) {
      polyline.back().emplace_back(tiles::latlng_to_fixed(pos));
    }
    f.geometry_ = polyline;

    std::lock_guard<std::mutex> lock(m_);
    f.id_ = seq_segs_.size();
    seq_segs_.push_back(seq_segs);

    feature_inserter_->insert(f);
  }

  void finish() {
    {
      motis::logging::scoped_timer timer("finish index and boxes");
      finish_index();
      finish_boxes();
    }
    {
      motis::logging::scoped_timer timer("tiles: prepare");
      feature_inserter_.reset(nullptr);
      tiles::prepare_tiles(*db_->db_handle_, *db_->pack_handle_, 10);
    }
  }

  void finish_index() {
    message_creator mc;

    std::sort(begin(seq_infos_), end(seq_infos_));
    auto const fbs_seq_infos = utl::to_vec(seq_infos_, [&mc](auto const& info) {
      auto const fbs_station_ids =
          utl::to_vec(std::get<0>(info), [&mc](auto const& station_id) {
            return mc.CreateSharedString(station_id);
          });
      auto const& fbs_classes = std::get<1>(info);
      return CreatePathSeqInfo(mc, mc.CreateVector(fbs_station_ids),
                               mc.CreateVector(fbs_classes), std::get<2>(info));
    });

    auto const fbs_tile_feature_infos =
        utl::to_vec(seq_segs_, [&mc](auto const& seq_segs) {
          auto const infos = utl::to_vec(seq_segs, [](auto const& seq_seg) {
            return TileFeatureInfo{seq_seg.sequence_, seq_seg.segment_};
          });
          return CreateTileFeatureInfos(mc, mc.CreateVectorOfStructs(infos));
        });

    mc.Finish(CreatePathIndex(mc, mc.CreateVector(fbs_seq_infos),
                              mc.CreateVector(fbs_tile_feature_infos)));

    using path_index = typed_flatbuffer<PathIndex>;
    db_put(kIndexKey, path_index(std::move(mc)).to_string());
  }

  void finish_boxes() {
    message_creator mc;
    mc.create_and_finish(
        MsgContent_PathBoxesResponse,
        CreatePathBoxesResponse(
            mc, mc.CreateVector(utl::to_vec(
                    boxes_,
                    [&mc](auto const& pair) {
                      auto ne = motis::Position{pair.second.max_.lat_,
                                                pair.second.max_.lng_};
                      auto sw = motis::Position{pair.second.min_.lat_,
                                                pair.second.min_.lng_};

                      return CreateBox(
                          mc, mc.CreateSharedString(pair.first.first),
                          mc.CreateSharedString(pair.first.second), &ne, &sw);
                    })))
            .Union());

    db_put(kBoxesKey, make_msg(mc)->to_string());
  }

  void db_put(std::string const& k, std::string const& v) const {
    auto txn = db_->db_handle_->make_txn();
    auto dbi = db_->data_dbi(txn);
    txn.put(dbi, k, v);
    txn.commit();
  }

  std::mutex m_;

  std::unique_ptr<path_database> db_;
  std::unique_ptr<tiles::feature_inserter_mt> feature_inserter_;

  size_t station_layer_id_;
  size_t path_layer_id_;

  std::vector<seq_info> seq_infos_;
  std::vector<std::vector<seq_seg>> seq_segs_;

  mcd::hash_map<std::pair<std::string, std::string>, geo::box> boxes_;
};

db_builder::db_builder(std::string const& fname)
    : impl_{std::make_unique<impl>(fname)} {}
db_builder::~db_builder() = default;

void db_builder::store_stations(std::vector<station> const& stations) const {
  impl_->store_stations(stations);
}

void db_builder::add_seq(
    size_t seq_idx, resolved_station_seq const& resolved_sequences,
    std::vector<processed_segment> const& processed_segments) const {
  impl_->add_seq(seq_idx, resolved_sequences, processed_segments);
}

void db_builder::add_tile_feature(geo::polyline const& polyline,
                                  std::vector<seq_seg> const& seq_segs,
                                  std::vector<uint32_t> const& classes,
                                  bool is_stub) const {
  impl_->add_tile_feature(polyline, seq_segs, classes, is_stub);
}

void db_builder::finish() const { impl_->finish(); }

}  // namespace motis::path
