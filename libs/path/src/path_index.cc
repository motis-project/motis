#include "motis/path/path_index.h"

#include <map>

#include "utl/to_vec.h"
#include "utl/verify.h"

#include "motis/path/error.h"

namespace motis::path {

path_index::path_index(std::string const& s)
    : path_index(index_t{s.size(), s.data()}) {}

path_index::path_index(index_t const index) {
  for (auto const& entry : *index.get()->sequences()) {
    auto const seq = utl::to_vec(*entry->station_ids(),
                                 [](auto const& id) { return id->str(); });
    utl::verify(!seq.empty(), "empty sequence!");
    for (auto const& clasz : *entry->classes()) {
      seq_map_.insert({{seq, service_class{clasz}}, entry->sequence()});
    }
  }

  seq_keys_.resize(index.get()->sequences()->size());
  for (auto const& seq : *index.get()->sequences()) {
    utl::verify(seq->sequence() < seq_keys_.size(),
                "invalid sequence idx found");
    seq_keys_.at(seq->sequence()) =
        seq_info{utl::to_vec(*seq->station_ids(),
                             [](auto const& str) { return str->str(); }),
                 utl::to_vec(*seq->classes(),
                             [](auto const& c) { return service_class{c}; })};
  }

  tile_features_ =
      utl::to_vec(*index.get()->tile_features(), [](auto const& infos) {
        return utl::to_vec(*infos->info(), [](auto const& info) {
          return std::make_pair(info->sequence(), info->segment());
        });
      });
}

size_t path_index::find(seq_key const& k) const {
  auto const it = seq_map_.find(k);
  utl::verify_ex(it != end(seq_map_),
                 std::system_error{error::unknown_sequence});
  return it->second;
}

// XXX verify ->  system_error
std::vector<path_index::segment_info> path_index::get_segments(
    size_t const ref) const {
  utl::verify(ref < tile_features_.size(), "invalid ref");

  std::map<std::pair<std::string, std::string>, std::set<service_class>> acc;
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
        std::vector<service_class>{begin(pair.second), end(pair.second)}};
  });
}

}  // namespace motis::path
