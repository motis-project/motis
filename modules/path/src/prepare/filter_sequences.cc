#include "motis/path/prepare/filter_sequences.h"

#include "boost/algorithm/string/classification.hpp"
#include "boost/algorithm/string/split.hpp"
#include "boost/filesystem.hpp"

#include "geo/polygon.h"

#include "utl/erase_if.h"
#include "utl/verify.h"

#include "motis/core/common/logging.h"

namespace ml = motis::logging;
namespace fs = boost::filesystem;

namespace motis::path {

void filter_sequences(std::vector<std::string> const& filters,
                      std::vector<station_seq>& sequences) {
  ml::scoped_timer timer("filter station sequences");
  for (auto const& filter : filters) {
    std::vector<std::string> tokens;
    boost::split(tokens, filter, boost::is_any_of(":"));
    utl::verify(tokens.size() == 2, "unexpected filter");

    if (tokens[0] == "id") {
      utl::erase_if(sequences, [&tokens](auto const& seq) {
        return std::none_of(
            begin(seq.station_ids_), end(seq.station_ids_),
            [&tokens](auto const& id) { return id == tokens[1]; });
      });
    } else if (tokens[0] == "seq") {
      std::vector<std::string> ids;
      boost::split(ids, tokens[1], boost::is_any_of("."));
      utl::erase_if(sequences, [&ids](auto const& seq) {
        return ids != seq.station_ids_;
      });
    } else if (tokens[0] == "extent") {
      utl::verify(fs::is_regular_file(tokens[1]), "cannot find extent polygon");
      auto const extent_polygon = geo::read_poly_file(tokens[1]);
      utl::erase_if(sequences, [&](auto const& seq) {
        return std::any_of(begin(seq.coordinates_), end(seq.coordinates_),
                           [&](auto const& coord) {
                             return !geo::within(coord, extent_polygon);
                           });
      });
    } else if (tokens[0] == "limit") {
      size_t const count = std::stoul(tokens[1]);
      sequences.resize(std::min(count, sequences.size()));
    } else if (tokens[0] == "cat") {
      auto cat = std::stoi(tokens[1]);
      utl::erase_if(sequences, [&](auto const& seq) {
        return seq.categories_.find(cat) == end(seq.categories_);
      });
      for (auto& seq : sequences) {
        seq.categories_ = {cat};
      }
    } else {
      LOG(ml::info) << "unknown filter: " << tokens[0];
    }
  }
}

}  // namespace motis::path