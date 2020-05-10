#pragma once

#include <map>
#include <string>

#include "motis/hash_map.h"

#include "utl/erase_duplicates.h"
#include "utl/to_vec.h"

#include "motis/core/common/typed_flatbuffer.h"

#include "motis/path/error.h"

#include "motis/path/fbs/PathIndex_generated.h"

namespace motis::path {

struct path_index {
  using class_t = uint32_t;
  using index_t = typed_flatbuffer<PathIndex>;

  struct seq_key {
    std::vector<std::string> station_ids_;
    class_t clasz_ = 0U;
  };

  struct seq_info {
    std::vector<std::string> station_ids_;
    std::vector<class_t> classes_;
  };

  path_index() = default;

  explicit path_index(std::string const& s)
      : path_index(index_t{s.size(), s.data()}) {}

  explicit path_index(index_t const& index) {
    for (auto const& entry : *index.get()->sequences()) {
      auto const seq = utl::to_vec(*entry->station_ids(),
                                   [](auto const& id) { return id->str(); });
      utl::verify(!seq.empty(), "empty sequence!");
      for (auto const& clasz : *entry->classes()) {
        seq_map_.insert({{seq, clasz}, entry->sequence()});
      }
    }

    seq_keys_.resize(index.get()->sequences()->size());
    for (auto const& seq : *index.get()->sequences()) {
      utl::verify(seq->sequence() < seq_keys_.size(),
                  "invalid sequence idx found");
      seq_keys_.at(seq->sequence()) =
          seq_info{utl::to_vec(*seq->station_ids(),
                               [](auto const& str) { return str->str(); }),
                   {seq->classes()->begin(), seq->classes()->end()}};
    }

    tile_features_ =
        utl::to_vec(*index.get()->tile_features(), [](auto const& infos) {
          return utl::to_vec(*infos->info(), [](auto const& info) {
            return std::make_pair(info->sequence(), info->segment());
          });
        });
  }

  size_t find(seq_key const& k) const {
    auto const it = seq_map_.find(k);
    if (it == end(seq_map_)) {
      throw std::system_error(error::unknown_sequence);
    }
    return it->second;
  }

  struct segment_info {
    std::string from_, to_;
    std::vector<class_t> classes_;
  };

  // XXX verify ->  system_error
  std::vector<segment_info> get_segments(size_t const ref) const {
    utl::verify(ref < tile_features_.size(), "invalid ref");

    std::map<std::pair<std::string, std::string>, std::set<class_t>> acc;
    for (auto const& pair : tile_features_[ref]) {
      utl::verify(pair.first < seq_keys_.size(), "invalid feature");
      auto const& key = seq_keys_[pair.first];

      utl::verify(pair.second < key.station_ids_.size() - 1, "invalid feature");

      acc[{key.station_ids_[pair.second], key.station_ids_[pair.second + 1]}]
          .insert(begin(key.classes_), end(key.classes_));
    }

    return utl::to_vec(acc, [](auto const& pair) {
      return segment_info{
          pair.first.first, pair.first.second,
          std::vector<class_t>{begin(pair.second), end(pair.second)}};
    });
  }

  mcd::hash_map<seq_key, size_t> seq_map_;

  std::vector<std::vector<std::pair<uint32_t, uint32_t>>> tile_features_;
  std::vector<seq_info> seq_keys_;
};

}  // namespace motis::path
