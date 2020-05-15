#include "motis/path/path_database_query.h"

#include "tiles/db/tile_index.h"
#include "tiles/fixed/convert.h"
#include "tiles/fixed/fixed_geometry.h"
#include "tiles/get_tile.h"

#include "utl/enumerate.h"
#include "utl/get_or_create.h"
#include "utl/join.h"
#include "utl/repeat_n.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "motis/path/constants.h"
#include "motis/path/polyline_builder.h"

using namespace flatbuffers;

namespace motis::path {

void path_database_query::add_sequence(size_t const index,
                                       std::vector<size_t> segment_indices) {
  sequences_.emplace_back(index, std::move(segment_indices));
}

void path_database_query::add_extra(std::vector<geo::polyline> extra) {
  resolvable_sequence rs{kExtraSequenceIndex, {}};
  for (auto const& line : extra) {
    auto rf = std::make_unique<resolvable_feature>();
    rf->is_resolved_ = true;
    rf->is_extra_ = true;
    rf->fwd_use_count_ = 1;

    tiles::fixed_polyline polyline;
    polyline.emplace_back();
    polyline.back().reserve(line.size());
    for (auto const& pos : line) {
      polyline.back().emplace_back(tiles::latlng_to_fixed(pos));
    }
    rf->geometry_ = polyline;

    rs.segment_features_.emplace_back();
    rs.segment_features_.back().emplace_back(
        true, extras_.emplace_back(std::move(rf)).get());
  }
  sequences_.emplace_back(std::move(rs));
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
    auto const layer_it = std::find(begin(render_ctx.layer_names_),
                                    end(render_ctx.layer_names_), "path");
    utl::verify(layer_it != end(render_ctx.layer_names_),
                "path_database_query::missing path layer");
    auto const layer_idx =
        std::distance(begin(render_ctx.layer_names_), layer_it);

    auto dbi = db.db_handle_->features_dbi(txn);
    auto cursor = lmdb::cursor{txn, dbi};
    for (auto& [hint, subquery] : subqueries_) {
      execute_subquery(hint, subquery, layer_idx, cursor, *db.pack_handle_,
                       render_ctx);
    }
  }
}

void path_database_query::resolve_sequences_and_build_subqueries(
    lmdb::cursor& cursor) {
  for (auto& rs : sequences_) {
    if (rs.index_ == kExtraSequenceIndex) {
      continue;
    }

    auto ret = cursor.get(lmdb::cursor_op::SET, std::to_string(rs.index_));
    utl::verify(ret.has_value(), "path_database_query: {} not found :E", index);
    auto const* ptr =
        flatbuffers::GetRoot<InternalDbSequence>(ret->second.data());

    utl::verify(ptr->classes()->size() != 0,
                "path_database_query: have empty classes");
    auto const min_clasz = *std::min_element(std::begin(*ptr->classes()),
                                             std::end(*ptr->classes()));

    auto const add_segment = [&](auto const* segment) {
      rs.segment_features_.emplace_back();

      if (segment->features()->size() == 0) {
        auto fallback = std::make_unique<resolvable_feature>();
        fallback->feature_id_ = std::numeric_limits<uint64_t>::max();
        fallback->fwd_use_count_ = 1;
        fallback->is_resolved_ = true;

        tiles::fixed_polyline geo;
        geo.emplace_back();
        geo.back().push_back(tiles::latlng_to_fixed(
            {segment->fallback_lat(), segment->fallback_lng()}));
        fallback->geometry_ = std::move(geo);

        rs.segment_features_.back().emplace_back(
            true, extras_.emplace_back(std::move(fallback)).get());
        return;
      }

      utl::verify(segment->hints_rle()->size() % 2 == 0, "invalid rle");
      size_t k = 0;

      for (auto i = 0ULL; i < segment->hints_rle()->size(); i += 2) {
        auto& subquery = subqueries_[segment->hints_rle()->Get(i)];

        for (auto j = 0ULL; j < segment->hints_rle()->Get(i + 1); ++j) {
          auto feature_id = segment->features()->Get(k);
          auto* resolvable =
              utl::get_or_create(subquery.map_, std::abs(feature_id), [&] {
                auto r = std::make_unique<resolvable_feature>();
                r->feature_id_ = std::abs(feature_id);
                return subquery.mem_.emplace_back(std::move(r)).get();
              });

          resolvable->min_clasz_ = std::min(resolvable->min_clasz_, min_clasz);

          auto const is_fwd = feature_id >= 0;
          if (is_fwd) {
            ++resolvable->fwd_use_count_;
          } else {
            ++resolvable->bwd_use_count_;
          }

          rs.segment_features_.back().emplace_back(is_fwd, resolvable);

          ++k;
        }
      }

      utl::verify(k == segment->features()->size(),
                  "path_database_query: features/hint_rle missmatch");
    };

    if (rs.segment_indices_.empty()) {
      for (auto const* segment : *ptr->segments()) {
        add_segment(segment);
      }
    } else {
      for (auto const segment_index : rs.segment_indices_) {
        utl::verify(segment_index < ptr->segments()->size(),
                    "path_database_query: invalid segment_index");
        add_segment(ptr->segments()->Get(segment_index));
      }
    }
  }
}

template <typename POD, size_t Stride = sizeof(POD)>
struct pod_mem_iterator {
  using self_t = pod_mem_iterator<POD, Stride>;

  using iterator_category = std::random_access_iterator_tag;
  using value_type = POD;
  using difference_type = int64_t;
  using pointer = POD*;
  using reference = POD&;

  pod_mem_iterator(char const* base, size_t index)
      : base_{base}, index_{index} {}

  POD operator*() const { return tiles::read<POD>(base_, index_ * Stride); };

  self_t& operator++() {
    ++index_;
    return *this;
  }

  int64_t operator-(self_t const& rhs) const { return index_ - rhs.index_; }

  bool operator==(self_t const& rhs) const {
    return std::tie(base_, index_) == std::tie(rhs.base_, rhs.index_);
  }

  bool operator!=(self_t const& rhs) const {
    return std::tie(base_, index_) != std::tie(rhs.base_, rhs.index_);
  }

  char const* base_;
  size_t index_;
};

// see: https://en.cppreference.com/w/cpp/algorithm/remove_copy
template <class InputIt, class OutputIt, class UnaryPredicate,
          class UnaryOperation>
OutputIt remove_transform_if(InputIt first, InputIt last, OutputIt d_first,
                             UnaryPredicate p, UnaryOperation u) {
  for (; first != last; ++first) {
    if (!p(*first)) {
      *d_first++ = u(*first);
    }
  }
  return d_first;
}

using resolvable_feature = path_database_query::resolvable_feature;
using resolvable_feature_ptr = std::unique_ptr<resolvable_feature>*;

template <typename Fn>
void unpack_features(
    std::string_view const pack,
    std::vector<std::pair<resolvable_feature_ptr,
                          resolvable_feature_ptr>> const& query_clasz_bounds,
    Fn&& fn) {
  utl::verify(pack.size() >= 5, "path_database_query: invalid feature_pack");
  auto const idx_offset = tiles::find_segment_offset(pack, kPathIndexId);
  utl::verify(idx_offset.has_value(), "path_database_query: index missing!");

  constexpr auto const kHeaderSize = kNumMotisClasses * sizeof(uint32_t);
  auto const header_base = pack.data() + *idx_offset;
  auto const indices_base = header_base + kHeaderSize;
  auto const pack_end = pack.data() + pack.size();

  using feature_it_t = pod_mem_iterator<uint32_t, 2 * sizeof(uint32_t)>;
  std::vector<std::pair<feature_it_t, feature_it_t>> index_clasz_bounds;

  size_t index = 0;
  for (auto i = 0; i < kNumMotisClasses; ++i) {
    auto const feature_count = tiles::read_nth<uint32_t>(header_base, i);
    index_clasz_bounds.emplace_back(
        feature_it_t{indices_base, index},
        feature_it_t{indices_base, index + feature_count});
    index += feature_count;
  }

  // invariants for active_queries set
  // - vector is always sorted by feature_id
  // - contains only unresolved features
  // - contains only features with min_clasz >= (current) clasz
  std::vector<resolvable_feature*> active_queries;
  for (int clasz = kNumMotisClasses - 1; clasz >= 0; --clasz) {
    {  // add new features from this zoomlevel
      auto const& [lb, ub] = query_clasz_bounds[clasz];
      if (lb != nullptr || lb != ub) {  // any new elements
        auto const size_old = active_queries.size();
        remove_transform_if(
            lb, ub, std::back_inserter(active_queries),
            [](auto const& f) { return f->is_resolved_; },
            [](auto const& f) { return f.get(); });

        // any old elements or any new elements -> "sort" the range
        if (size_old != 0 && size_old != active_queries.size()) {
          std::inplace_merge(
              begin(active_queries), std::next(begin(active_queries), size_old),
              end(active_queries), [](auto const* lhs, auto const* rhs) {
                return lhs->feature_id_ < rhs->feature_id_;
              });
        }
      }
    }

    if (active_queries.empty()) {
      continue;
    }

    struct join_key {
      auto key(uint32_t const id) { return id; }
      auto key(resolvable_feature const* f) { return f->feature_id_; }
    };
    utl::inner_join(
        index_clasz_bounds[clasz].first, index_clasz_bounds[clasz].second,
        begin(active_queries), end(active_queries), utl::join_less<join_key>{},
        [&](auto idx_lb, auto idx_ub, auto q_lb, auto q_ub) {
          utl::verify(std::distance(idx_lb, idx_ub) == 1,
                      "path_database_query: duplicate in index");
          utl::verify(std::distance(q_lb, q_ub) == 1,
                      "path_database_query: duplicate in query");

          auto const feature_offset =
              tiles::read_nth<uint32_t>(indices_base, idx_lb.index_ * 2 + 1);

          auto feature_ptr = pack.data() + feature_offset;
          auto feature_size = protozero::decode_varint(&feature_ptr, pack_end);

          fn(*q_lb, std::string_view{feature_ptr, feature_size});
        });

    utl::erase_if(active_queries,
                  [](auto const* f) { return f->is_resolved_; });
  }
}

void path_database_query::execute_subquery(
    tiles::tile_index_t const hint, subquery& q, size_t const layer_idx,
    lmdb::cursor& cursor, tiles::pack_handle const& pack_handle,
    tiles::render_ctx const& render_ctx) {
  std::sort(begin(q.mem_), end(q.mem_), [](auto const& lhs, auto const& rhs) {
    return std::tie(lhs->min_clasz_, lhs->feature_id_) <
           std::tie(rhs->min_clasz_, rhs->feature_id_);
  });

  std::vector<std::pair<resolvable_feature_ptr, resolvable_feature_ptr>>
      query_clasz_bounds(kNumMotisClasses, {nullptr, nullptr});
  utl::equal_ranges_linear(
      q.mem_,
      [](auto const& lhs, auto const& rhs) {
        return lhs->min_clasz_ == rhs->min_clasz_;
      },
      [&](auto lb, auto ub) {
        utl::verify((**lb).min_clasz_ < kNumMotisClasses, "invalid min_clasz");
        query_clasz_bounds[(**lb).min_clasz_] =
            std::make_pair(&*lb, &*lb + std::distance(lb, ub));
      });

  auto const tile = tiles::tile_key_to_tile(hint);
  tiles::pack_records_foreach(cursor, tile, [&](auto, auto pack_record) {
    unpack_features(
        pack_handle.get(pack_record), query_clasz_bounds,
        [&](resolvable_feature* rf, std::string_view const feature_str) {
          // TODO deserialize _only geometry_
          auto const feature = deserialize_feature(
              feature_str, render_ctx.metadata_decoder_,
              tiles::fixed_box{
                  {tiles::kInvalidBoxHint, tiles::kInvalidBoxHint},
                  {tiles::kInvalidBoxHint, tiles::kInvalidBoxHint}},
              tiles::kInvalidZoomLevel);
          // zoom_level_ < 0 ? tiles::kInvalidZoomLevel : zoom_level_);

          utl::verify(feature.has_value(), "deserialize failed");
          utl::verify(feature->layer_ == layer_idx, "wrong layer");

          rf->geometry_ = std::move(feature->geometry_);
          rf->is_resolved_ = true;
        });
  });
}

Offset<PathSeqResponse> path_database_query::write_sequence(
    module::message_creator& mc, path_database const& db, size_t const index) {
  auto const it =
      std::find_if(begin(sequences_), end(sequences_),
                   [&](auto const& rs) { return rs.index_ == index; });
  utl::verify(it != end(sequences_), "path_database_query: write unknown seq");

  utl::verify(it->segment_indices_.empty(),
              "path_database_query: partial sequence export not implemented!");

  auto txn = db.db_handle_->make_txn();
  auto dbi = path_database::data_dbi(txn);

  auto ret = txn.get(dbi, std::to_string(index));
  utl::verify(ret.has_value(), "path_database_query: {} not found :W", index);

  auto const* ptr = flatbuffers::GetRoot<InternalDbSequence>(ret->data());

  auto const station_ids =
      utl::to_vec(*ptr->station_ids(),
                  [&mc](auto const& e) { return mc.CreateString(e->c_str()); });
  auto const classes = utl::to_vec(*ptr->classes());

  double_polyline_builder pb;
  std::vector<Offset<Segment>> fbs_segments;
  for (auto const& [i, s_features] : utl::enumerate(it->segment_features_)) {
    pb.clear();
    for (auto const& [is_fwd, rf] : s_features) {
      utl::verify(rf->is_resolved_,
                  "path_database_query: have unresolved feature! {} {}",
                  rf->feature_id_, rf->min_clasz_);

      pb.append(is_fwd, rf);
    }
    pb.finish();

    auto osm_node_ids = utl::repeat_n(int64_t{-1}, pb.coords_.size() / 2);

    fbs_segments.emplace_back(CreateSegment(mc, mc.CreateVector(pb.coords_),
                                            mc.CreateVector(osm_node_ids)));
  }

  return CreatePathSeqResponse(
      mc, mc.CreateVector(station_ids), mc.CreateVector(classes),
      mc.CreateVector(fbs_segments),
      mc.CreateVector(std::vector<Offset<PathSourceInfo>>{}));
}

Offset<PathByTripIdBatchResponse> path_database_query::write_batch(
    module::message_creator& mc) {
  std::vector<Offset<PolylineIndices>> fbs_segments;
  std::vector<Offset<String>> fbs_polylines;
  std::vector<size_t> fbs_extras;

  // "disallow" id zero -> cant be marked reversed (-0 == 0)
  fbs_polylines.emplace_back(mc.CreateString(std::string{}));

  google_polyline_builder pb;
  auto const finish_polyline = [&] {
    pb.finish();
    auto const index = fbs_polylines.size();
    fbs_polylines.emplace_back(mc.CreateString(pb.enc_.buf_));
    if (pb.is_extra_) {
      fbs_extras.push_back(index);
    }
    pb.clear();
    return index;
  };

  {  // write multi-used polylines first
    std::vector<resolvable_feature*> multi_use;
    for (auto const& [idx, subquery] : subqueries_) {
      for (auto const& rf : subquery.mem_) {
        if (rf->use_count() > 1) {
          multi_use.emplace_back(rf.get());
        }
      }
    }
    std::sort(begin(multi_use), end(multi_use),
              [](auto const* lhs, auto const* rhs) {
                auto const lhs_count = lhs->use_count();
                auto const rhs_count = rhs->use_count();
                return std::tie(lhs_count, lhs->feature_id_) <
                       std::tie(rhs_count, rhs->feature_id_);
              });
    for (auto it = rbegin(multi_use); it != rend(multi_use); ++it) {
      (**it).is_reversed_ = (**it).bwd_use_count_ > (**it).fwd_use_count_;
      pb.append(!(**it).is_reversed_, *it);
      (**it).response_id_ = finish_polyline();
    }
  }

  // write segments (and single-used polylines)
  for (auto const& sequence : sequences_) {
    for (auto const& segment : sequence.segment_features_) {
      std::vector<int64_t> indices;
      for (auto const& [is_fwd, rf] : segment) {
        if (rf->response_id_ != kInvalidResponseId) {
          if (!pb.empty()) {
            indices.push_back(finish_polyline());
          }

          // if segment requirement and feature state dont match -> negate id
          indices.push_back(static_cast<int64_t>(rf->response_id_) *
                            ((is_fwd != rf->is_reversed_) ? 1 : -1));
        } else {
          utl::verify(rf->use_count() == 1, "multi use slipped through");
          utl::verify(!rf->is_reversed_, "have reversed single use");

          pb.append(is_fwd, rf);
        }
      }

      if (!pb.empty()) {
        indices.push_back(finish_polyline());
      }

      fbs_segments.emplace_back(
          CreatePolylineIndices(mc, mc.CreateVector(indices)));
    }
  }

  return CreatePathByTripIdBatchResponse(mc, mc.CreateVector(fbs_segments),
                                         mc.CreateVector(fbs_polylines),
                                         mc.CreateVector(fbs_extras));
}

}  // namespace motis::path
