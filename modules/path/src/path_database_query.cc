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
          auto* resolvable =
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

      utl::verify(k == segment->features()->size(),
                  "path_database_query: features/hint_rle missmatch");
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
  for (auto const& [i, s_features] : utl::enumerate(it->segment_features_)) {
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

    for (auto const& [is_fwd, resolvable] : s_features) {
      utl::verify(resolvable->is_resolved_,
                  "path_database_query: have unresolved feature! {} {}",
                  resolvable->feature_id_, resolvable->min_clasz_);

      auto const& l = mpark::get<tiles::fixed_polyline>(resolvable->geometry_);
      utl::verify(l.size() == 1 && l.front().size() > 1,
                  "invalid line geometry");
      if (is_fwd) {
        std::for_each(std::begin(l.front()), std::end(l.front()), append);
      } else {
        std::for_each(std::rbegin(l.front()), std::rend(l.front()), append);
      }
    }

    if (coordinates.empty()) {
      coordinates.push_back(ptr->segments()->Get(i)->fallback_lat());
      coordinates.push_back(ptr->segments()->Get(i)->fallback_lng());
      coordinates.push_back(ptr->segments()->Get(i)->fallback_lat());
      coordinates.push_back(ptr->segments()->Get(i)->fallback_lng());
    }

    auto osm_node_ids = utl::repeat_n(int64_t{-1}, coordinates.size() / 2);

    utl::verify(coordinates.size() >= 4, "empty coordinates {}",
                coordinates.size());
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
