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
#include "utl/zip.h"

#include "tiles/db/feature_inserter_mt.h"
#include "tiles/db/layer_names.h"
#include "tiles/db/prepare_tiles.h"
#include "tiles/db/tile_database.h"
#include "tiles/feature/feature.h"
#include "tiles/fixed/convert.h"
#include "tiles/fixed/fixed_geometry.h"
#include "tiles/fixed/io/serialize.h"

#include "motis/hash_map.h"
#include "motis/hash_set.h"

#include "motis/core/common/logging.h"
#include "motis/core/schedule/connection.h"
#include "motis/module/message.h"

#include "motis/path/definitions.h"
#include "motis/path/path_database.h"
#include "motis/path/path_zoom_level.h"
#include "motis/path/prepare/db_tiles_packer.h"

#include "motis/path/fbs/InternalDbSequence_generated.h"
#include "motis/path/fbs/PathIndex_generated.h"
#include "motis/protocol/PathSeqResponse_generated.h"

using namespace flatbuffers;
using namespace motis::module;

namespace motis::path {

using seq_info =
    std::tuple<mcd::vector<mcd::string>, mcd::vector<service_class>, int>;

template <typename Classes>
service_class get_min_clasz(Classes const& c) {
  if (c.empty()) {
    return service_class::OTHER;
  }
  return *std::min_element(begin(c), end(c));
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

    seq_segs_.emplace_back();  // dont use id zero (-0 == 0)
  }

  impl(impl const&) = delete;
  impl& operator=(impl const&) = delete;
  impl(impl&&) noexcept = delete;
  impl& operator=(impl&&) noexcept = delete;

  ~impl() {
    utl::verify(db_cache_size_ == 0 && db_cache_.empty(),
                "db_builder: cache is not empty in dtor");
  }

  void store_stations(mcd::vector<station_seq> const& sequences) {
    struct station_info {
      mcd::string name_;
      geo::latlng pos_;
      int min_z_{tiles::kMaxZoomLevel};
      service_class min_clasz_{service_class::OTHER};
    };

    mcd::hash_map<mcd::string, station_info> station_infos;
    for (auto const& seq : sequences) {
      auto const min_z = std::accumulate(
          begin(seq.classes_), end(seq.classes_), min_zoom_level(),
          [&](auto z, auto c) {
            return std::min(z, min_zoom_level(c, seq.distance_));
          });
      auto const min_clasz = get_min_clasz(seq.classes_);

      for (auto const& [id, name, pos] :
           utl::zip(seq.station_ids_, seq.station_names_, seq.coordinates_)) {
        auto const& [it, new_elem] = station_infos.emplace(
            id, station_info{name, pos, min_z, min_clasz});
        if (!new_elem) {
          it->second.min_z_ = std::min(it->second.min_z_, min_z);
          it->second.min_clasz_ = std::min(it->second.min_clasz_, min_clasz);
        }
      }
    }

    for (auto const& [id, info] : station_infos) {
      tiles::feature f;
      f.id_ = station_feature_id_++;
      f.layer_ = station_layer_id_;
      f.zoom_levels_ = {info.min_z_, tiles::kMaxZoomLevel};

      f.meta_.emplace_back("id", tiles::encode_string(id));
      f.meta_.emplace_back("name", tiles::encode_string(info.name_));
      f.meta_.emplace_back(
          "min_class",
          tiles::encode_integer(static_cast<service_class_t>(info.min_clasz_)));

      f.geometry_ = tiles::fixed_point{
          {tiles::latlng_to_fixed({info.pos_.lat_, info.pos_.lng_})}};

      feature_inserter_->insert(f);
    }
  }

  std::pair<uint64_t, uint64_t> add_feature(
      geo::polyline const& line, std::vector<seq_seg> const& seq_segs,
      mcd::vector<service_class> const& classes, bool const is_stub,
      float const distance) {
    auto const min_z = std::accumulate(
        begin(classes), end(classes), min_zoom_level(), [&](auto z, auto c) {
          return std::min(z, min_zoom_level(c, distance));
        });
    auto const min_clasz = get_min_clasz(classes);

    tiles::feature f;
    f.layer_ = path_layer_id_;
    f.zoom_levels_ = {min_z, tiles::kMaxZoomLevel};

    f.meta_.emplace_back(
        "min_class",
        tiles::encode_integer(static_cast<service_class_t>(min_clasz)));
    f.meta_.emplace_back("stub", tiles::encode_bool(is_stub));

    tiles::fixed_polyline polyline;
    polyline.emplace_back();
    polyline.back().reserve(line.size());
    for (auto const& pos : line) {
      polyline.back().emplace_back(tiles::latlng_to_fixed(pos));
    }
    f.geometry_ = polyline;

    auto const lock = std::lock_guard{m_};
    f.id_ = seq_segs_.size();
    seq_segs_.push_back(seq_segs);

    auto const tile = feature_inserter_->insert(f);
    return {f.id_, tiles::tile_to_key(tile)};
  }

  void add_seq(size_t seq_idx, station_seq const& seq,
               std::vector<geo::box> const& boxes,
               std::vector<std::vector<int64_t>> const& feature_ids,
               std::vector<std::vector<uint64_t>> const& hints_rle) {
    utl::verify(boxes.size() + 1 == seq.station_ids_.size() &&
                    feature_ids.size() + 1 == seq.station_ids_.size() &&
                    hints_rle.size() + 1 == seq.station_ids_.size(),
                "add_seq: size mismatch");

    message_creator mc;
    {
      auto const fbs_stations =
          utl::to_vec(seq.station_ids_,
                      [&](auto const& id) { return mc.CreateString(id); });

      std::vector<Offset<InternalDbSegment>> fbs_segments;
      for (auto i = 0UL; i < feature_ids.size(); ++i) {
        auto const& original = seq.paths_.at(i).polyline_;
        utl::verify(!original.empty(), "add_seq: empty original polyline");

        fbs_segments.emplace_back(CreateInternalDbSegment(
            mc, mc.CreateVector(feature_ids[i]), mc.CreateVector(hints_rle[i]),
            original.front().lat_, original.front().lng_));
      }

      mc.Finish(CreateInternalDbSequence(
          mc, mc.CreateVector(fbs_stations),
          mc.CreateVector(utl::to_vec(
              seq.classes_,
              [](auto const& c) { return static_cast<service_class_t>(c); })),
          mc.CreateVector(fbs_segments)));
    }

    auto const lock = std::lock_guard{m_};
    update_boxes(seq.station_ids_, boxes);

    db_put(std::to_string(seq_idx),
           typed_flatbuffer<InternalDbSequence>{std::move(mc)}.to_string());
    seq_infos_.emplace_back(seq.station_ids_, seq.classes_, seq_idx);

    db_flush_maybe();
  }

  void update_boxes(mcd::vector<mcd::string> const& station_ids,
                    std::vector<geo::box> const& boxes) {
    for (auto i = 0UL; i < boxes.size(); ++i) {
      auto key =
          (station_ids[i] < station_ids[i + 1])
              ? std::pair{station_ids[i].str(), station_ids[i + 1].str()}
              : std::pair{station_ids[i + 1].str(), station_ids[i].str()};

      utl::get_or_create(boxes_, key, [] {
        return geo::box{};
      }).extend(boxes[i]);
    }
  }

  void finish() {
    {
      motis::logging::scoped_timer timer("finish index and boxes");
      finish_index();
      finish_boxes();
      db_flush_maybe(0);
    }
    feature_inserter_.reset(nullptr);

    auto progress_tracker = utl::get_active_progress_tracker();
    {
      motis::logging::scoped_timer timer("tiles: pack");
      progress_tracker->status("Pack Database").out_bounds(90, 95);

      auto const metadata_coder = make_shared_metadata_coder(*db_->db_handle_);
      pack_features(*db_->db_handle_, *db_->pack_handle_,
                    [&](auto const tile, auto const& packs) {
                      db_tiles_packer p{tile, metadata_coder, path_layer_id_};
                      p.pack_features(packs);
                      p.make_index();
                      return p.packer_.buf_;
                    });
    }
    {
      motis::logging::scoped_timer timer("tiles: prepare");
      progress_tracker->status("Prepare Tiles").out_bounds(95, 100);

      tiles::prepare_tiles(*db_->db_handle_, *db_->pack_handle_, 10);
    }
  }

  void finish_index() {
    message_creator mc;

    std::sort(begin(seq_infos_), end(seq_infos_));
    auto const fbs_seq_infos = utl::to_vec(seq_infos_, [&mc](auto const& info) {
      auto const fbs_station_ids =
          utl::to_vec(std::get<0>(info), [&mc](auto const& station_id) {
            return mc.CreateSharedString(station_id.data(), station_id.size());
          });
      auto const& fbs_classes = utl::to_vec(
          std::get<1>(info),
          [](auto const& c) { return static_cast<service_class_t>(c); });
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
                          mc,
                          mc.CreateSharedString(pair.first.first.data(),
                                                pair.first.first.size()),
                          mc.CreateSharedString(pair.first.second.data(),
                                                pair.first.second.size()),
                          &ne, &sw);
                    })))
            .Union());

    db_put(kBoxesKey, make_msg(mc)->to_string());
  }

  void db_put(std::string k, std::string v) {
    db_cache_size_ += v.size();
    db_cache_.emplace_back(std::move(k), std::move(v));
  }

  void db_flush_maybe(size_t min_cache_size = 128ULL * 1024 * 1024) {
    if (db_cache_size_ < min_cache_size) {
      return;
    }
    auto txn = db_->db_handle_->make_txn();
    auto dbi = db_->data_dbi(txn);
    for (auto const& [k, v] : db_cache_) {
      txn.put(dbi, k, v);
    }
    txn.commit();

    db_cache_size_ = 0;
    db_cache_.clear();
  }

  std::mutex m_;

  std::unique_ptr<path_database> db_;
  std::unique_ptr<tiles::feature_inserter_mt> feature_inserter_;

  size_t station_layer_id_;
  size_t path_layer_id_;

  size_t station_feature_id_{0};
  std::vector<seq_info> seq_infos_;
  std::vector<std::vector<seq_seg>> seq_segs_;

  size_t db_cache_size_{0};
  std::vector<std::pair<std::string, std::string>> db_cache_;

  mcd::hash_map<std::pair<std::string, std::string>, geo::box> boxes_;
};

db_builder::db_builder(std::string const& fname)
    : impl_{std::make_unique<impl>(fname)} {}
db_builder::~db_builder() = default;

void db_builder::store_stations(
    mcd::vector<station_seq> const& sequences) const {
  impl_->store_stations(sequences);
}

std::pair<uint64_t, uint64_t> db_builder::add_feature(
    geo::polyline const& polyline, std::vector<seq_seg> const& seq_segs,
    mcd::vector<service_class> const& classes, bool is_stub,
    float const distance) const {
  return impl_->add_feature(polyline, seq_segs, classes, is_stub, distance);
}

void db_builder::add_seq(
    size_t seq_idx, station_seq const& resolved_sequences,
    std::vector<geo::box> const& boxes,
    std::vector<std::vector<int64_t>> const& feature_ids,
    std::vector<std::vector<uint64_t>> const& hints_rle) const {
  impl_->add_seq(seq_idx, resolved_sequences, boxes, feature_ids, hints_rle);
}

void db_builder::finish() const { impl_->finish(); }

}  // namespace motis::path
