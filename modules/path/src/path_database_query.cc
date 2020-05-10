#include "motis/path/path_database_query.h"

#include "tiles/db/tile_index.h"
#include "tiles/fixed/convert.h"
#include "tiles/fixed/fixed_geometry.h"
#include "tiles/get_tile.h"

#include "utl/get_or_create.h"
#include "utl/repeat_n.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

using namespace flatbuffers;

namespace motis::path {

void path_database_query::add_sequence(size_t const index) {
  sequences_.push_back({index, {}});
}

void path_database_query::execute(path_database const& db,
                                  tiles::render_ctx const& render_ctx) {
  auto txn = db.db_handle_->make_txn();
  {
    auto dbi = path_database::data_dbi(txn);
    auto cursor = lmdb::cursor{txn, dbi};
    resolve_sequences_and_build_subqueries(cursor);
  }
  {
    auto dbi = db.db_handle_->features_dbi(txn);
    auto cursor = lmdb::cursor{txn, dbi};
    for (auto& [hint, subquery] : subqueries_) {
      execute_subquery(hint, subquery, cursor, *db.pack_handle_, render_ctx);
    }
  }
}

void path_database_query::resolve_sequences_and_build_subqueries(
    lmdb::cursor& cursor) {
  for (auto& [index, segment_features] : sequences_) {
    auto ret = cursor.get(lmdb::cursor_op::SET, std::to_string(index));
    utl::verify(ret.has_value(), "path_database_query: {} not found :E", index);
    auto const* ptr =
        flatbuffers::GetRoot<InternalDbSequence>(ret->second.data());

    utl::verify(ptr->classes()->size() != 0,
                "path_database_query: have empty classes");
    auto const min_clasz = *std::min_element(std::begin(*ptr->classes()),
                                             std::end(*ptr->classes()));

    for (auto const* segment : *ptr->segments()) {
      segment_features.emplace_back();

      utl::verify(segment->hints_rle()->size() % 2 == 0, "invalid rle");
      size_t k = 0;

      for (auto i = 0ULL; i < segment->hints_rle()->size(); i += 2) {
        auto& subquery = subqueries_[segment->hints_rle()->Get(i)];

        for (auto j = 0ULL; j < segment->hints_rle()->Get(i + 1); ++j) {
          auto feature_id = segment->features()->Get(k);
          auto& resolvable =
              utl::get_or_create(subquery.map_, std::abs(feature_id), [&] {
                auto r = std::make_unique<resolvable_feature>();
                r->feature_id_ = std::abs(feature_id);
                return subquery.mem_.emplace_back(std::move(r)).get();
              });

          resolvable->min_clasz_ = std::min(resolvable->min_clasz_, min_clasz);
          ++resolvable->include_count_;

          segment_features.back().emplace_back(feature_id >= 0, resolvable);

          ++k;
        }
      }
    }
  }
}

void path_database_query::execute_subquery(
    tiles::tile_index_t const hint, subquery& q, lmdb::cursor& cursor,
    tiles::pack_handle const& pack_handle,
    tiles::render_ctx const& render_ctx) {
  std::sort(begin(q.mem_), end(q.mem_), [](auto const& lhs, auto const& rhs) {
    return lhs->feature_id_ < rhs->feature_id_;
  });

  // TODO do this once for all subqueries
  auto const layer_it = std::find(begin(render_ctx.layer_names_),
                                  end(render_ctx.layer_names_), "path");
  utl::verify(layer_it != end(render_ctx.layer_names_),
              "path_database_query::missing path layer");
  auto const layer_idx =
      std::distance(begin(render_ctx.layer_names_), layer_it);

  auto const tile = tiles::tile_key_to_tile(hint);
  tiles::pack_records_foreach(cursor, tile, [&](auto, auto pack_record) {
    tiles::unpack_features(
        pack_handle.get(pack_record), [&](auto const& feature_str) {
          auto const feature = deserialize_feature(
              feature_str, render_ctx.metadata_decoder_,
              tiles::fixed_box{
                  {tiles::kInvalidBoxHint, tiles::kInvalidBoxHint},
                  {tiles::kInvalidBoxHint, tiles::kInvalidBoxHint}},
              tiles::kInvalidZoomLevel);
          // zoom_level_ < 0 ? tiles::kInvalidZoomLevel : zoom_level_);

          utl::verify(feature.has_value(), "deserialize failed");

          if (feature->layer_ != layer_idx) {
            return;  // wrong layer
          }

          auto it = std::lower_bound(begin(q.mem_), end(q.mem_), feature->id_,
                                     [](auto const& lhs, auto const& rhs) {
                                       return lhs->feature_id_ < rhs;
                                     });
          if (it == end(q.mem_) || (**it).feature_id_ != feature->id_) {
            return;  // not needed
          }

          (**it).geometry_ = std::move(feature->geometry_);
          (**it).is_resoved_ = true;
        });
  });
}

Offset<PathSeqResponse> path_database_query::write_sequence(
    module::message_creator& mc, path_database const& db, size_t const index) {
  auto const it =
      std::find_if(begin(sequences_), end(sequences_),
                   [&](auto const& rs) { return rs.index_ == index; });
  utl::verify(it != end(sequences_), "path_database_query: write unknown seq");

  auto txn = db.db_handle_->make_txn();
  auto dbi = path_database::data_dbi(txn);

  auto ret = txn.get(dbi, std::to_string(index));
  utl::verify(ret.has_value(), "path_database_query: {} not found :W", index);

  auto const* ptr = flatbuffers::GetRoot<InternalDbSequence>(ret->data());

  auto const station_ids =
      utl::to_vec(*ptr->station_ids(),
                  [&mc](auto const& e) { return mc.CreateString(e->c_str()); });
  auto const classes = utl::to_vec(*ptr->classes());

  std::vector<Offset<Segment>> fbs_segments;
  for (auto const& segment_features : it->segment_features_) {
    std::vector<double> coordinates;
    auto const append = [&](auto const& p) {
      auto const ll = tiles::fixed_to_latlng(p);
      if (coordinates.empty() ||
          coordinates[coordinates.size() - 2] != ll.lat_ ||
          coordinates[coordinates.size() - 1] != ll.lng_) {
        coordinates.push_back(ll.lat_);
        coordinates.push_back(ll.lng_);
      }
    };

    for (auto const& [is_fwd, resolvable] : segment_features) {
      utl::verify(resolvable->is_resoved_,
                  "path_database_query: have unresolved feature!");

      auto const& l = mpark::get<tiles::fixed_polyline>(resolvable->geometry_);
      utl::verify(l.size() == 1 && l.front().size() > 1,
                  "invalid line geometry");
      if (is_fwd) {
        std::for_each(std::begin(l.front()), std::end(l.front()), append);
      } else {
        std::for_each(std::rbegin(l.front()), std::rend(l.front()), append);
      }
    }

    auto osm_node_ids = utl::repeat_n(int64_t{-1}, coordinates.size() / 2);

    // TODO handle empty segments -> invent something based on station id
    utl::verify(coordinates.size() >= 2, "empty coordinates");
    if (coordinates.size() == 2) {
      coordinates.push_back(coordinates[0]);
      coordinates.push_back(coordinates[1]);
      osm_node_ids.push_back(-1);
    }

    fbs_segments.emplace_back(CreateSegment(mc, mc.CreateVector(coordinates),
                                            mc.CreateVector(osm_node_ids)));
  }

  return CreatePathSeqResponse(
      mc, mc.CreateVector(station_ids), mc.CreateVector(classes),
      mc.CreateVector(fbs_segments),
      mc.CreateVector(std::vector<Offset<PathSourceInfo>>{}));
}

Offset<PathSeqResponse> path_database_query::write_batch(
    module::message_creator&) {
  return {};
}

}  // namespace motis::path
