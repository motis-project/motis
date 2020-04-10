#include "motis/bikesharing/nextbike_initializer.h"

#include <functional>
#include <memory>

#include "boost/algorithm/string/predicate.hpp"
#include "boost/filesystem.hpp"

#include "pugixml.hpp"

#include "utl/to_vec.h"

#include "utl/parser/arg_parser.h"
#include "utl/parser/file.h"

#include "motis/core/common/constants.h"
#include "motis/core/common/logging.h"
#include "motis/module/context/motis_call.h"
#include "motis/module/message.h"
#include "motis/bikesharing/error.h"
#include "motis/bikesharing/geo.h"

using namespace flatbuffers;
using namespace motis::lookup;
using namespace motis::geo_detail;
using namespace motis::logging;
using namespace motis::module;
using namespace utl;
using namespace pugi;
namespace fs = boost::filesystem;
using fs::directory_iterator;

namespace motis::bikesharing {

void initialize_nextbike(std::string const& nextbike_path, database& db) {
  std::vector<terminal> terminals;
  std::vector<hourly_availabilities> availabilities;
  std::tie(terminals, availabilities) = load_and_merge(nextbike_path);

  auto const close_stations = find_close_stations(terminals);
  auto const close_terminals = find_close_terminals(terminals);

  std::vector<persistable_terminal> p;
  for (size_t i = 0; i < terminals.size(); ++i) {
    p.push_back(convert_terminal(terminals[i], availabilities[i],
                                 close_stations[i], close_terminals[i]));
  }
  db.put(p);
  db.put_summary(make_summary(terminals));
}

std::pair<std::vector<terminal>, std::vector<hourly_availabilities>>
load_and_merge(std::string const& nextbike_path) {
  auto files = get_nextbike_files(nextbike_path);
  LOG(info) << "loading " << files.size() << " NEXTBIKE XML files";
  if (files.empty()) {
    throw std::system_error(error::init_error);
  }

  manual_timer parse_timer("NEXTBIKE parsing");
  snapshot_merger merger;
  for (auto const& filename : files) {
    auto xml_string = utl::file(filename.c_str(), "r").content();
    auto timestamp = nextbike_filename_to_timestamp(filename);
    auto snapshot = nextbike_parse_xml(std::move(xml_string));
    merger.add_snapshot(timestamp, snapshot);
  }
  parse_timer.stop_and_print();

  scoped_timer merge_timer("NEXTBIKE merging");
  return merger.merged();
}

std::vector<std::string> get_nextbike_files(std::string const& path) {
  fs::path b{path};
  if (!fs::exists(b)) {
    return {};
  } else if (fs::is_regular_file(b)) {
    return {b.string()};
  } else if (fs::is_directory(b)) {
    std::vector<std::string> files;
    for (auto it = directory_iterator(b); it != directory_iterator(); ++it) {
      if (!fs::is_regular_file(it->status())) {
        continue;
      }

      auto filename = it->path().string();
      if (boost::algorithm::iends_with(filename, ".xml")) {
        files.push_back(filename);
      }
    }
    return files;
  }
  return {};
}

std::time_t nextbike_filename_to_timestamp(std::string const& filename) {
  auto dash_pos = filename.rfind('-');
  auto dot_pos = filename.rfind('.');
  if (dash_pos == std::string::npos || dot_pos == std::string::npos ||
      dash_pos > dot_pos) {
    throw std::runtime_error("unexpected nextbike filename");
  }

  return std::stoul(filename.substr(dash_pos + 1, dot_pos - dash_pos - 1));
}

std::vector<terminal_snapshot> nextbike_parse_xml(utl::buffer&& buf) {
  std::vector<terminal_snapshot> result;

  xml_document d;
  d.load_buffer_inplace(reinterpret_cast<void*>(buf.data()), buf.size());

  constexpr auto q = "/markers/country[@country='DE']/city/place[@spot='1']";
  for (auto const& xnode : d.select_nodes(q)) {
    auto const& node = xnode.node();

    terminal_snapshot terminal;
    terminal.uid_ = node.attribute("uid").value();
    terminal.lat_ = node.attribute("lat").as_double();
    terminal.lng_ = node.attribute("lng").as_double();
    terminal.name_ = node.attribute("name").value();
    terminal.available_bikes_ = parse<int>(node.attribute("bikes").value(), 0);
    result.push_back(terminal);
  }

  return result;
}

close_locations find_close_stations(std::vector<terminal> const& terminals) {
  auto const req = to_geo_request(terminals, MAX_WALK_DIST);
  auto const msg = motis_call(req)->val();

  using lookup::LookupBatchGeoStationResponse;
  auto const resp = motis_content(LookupBatchGeoStationResponse, msg);

  close_locations attached_stations;
  for (size_t i = 0; i < terminals.size(); ++i) {
    auto const& t = terminals[i];
    auto const& found_stations = resp->responses()->Get(i)->stations();
    attached_stations.push_back(
        utl::to_vec(*found_stations, [&t](auto&& stations) -> close_location {
          int const dist = distance_in_m(t.lat_, t.lng_, stations->pos()->lat(),
                                         stations->pos()->lng());
          int const dur = dist * LINEAR_DIST_APPROX / WALK_SPEED;
          return {stations->id()->str(), dur};
        }));
  }

  return attached_stations;
}

msg_ptr to_geo_request(std::vector<terminal> const& terminals, double r) {
  message_creator b;
  std::vector<Offset<lookup::LookupGeoStationRequest>> c;
  for (auto const& merged : terminals) {
    Position pos(merged.lat_, merged.lng_);
    c.push_back(CreateLookupGeoStationRequest(b, &pos, 0.0, r));
  }
  b.create_and_finish(
      MsgContent_LookupBatchGeoStationRequest,
      CreateLookupBatchGeoStationRequest(b, b.CreateVector(c)).Union(),
      "/lookup/geo_station_batch");
  return make_msg(b);
}

close_locations find_close_terminals(std::vector<terminal> const& terminals) {
  std::vector<value> values;
  for (size_t i = 0; i < terminals.size(); ++i) {
    values.emplace_back(spherical_point(terminals[i].lng_, terminals[i].lat_),
                        i);
  }
  bgi::rtree<value, bgi::quadratic<16>> rtree{values};

  close_locations all_close_terminals;

  for (size_t i = 0; i < terminals.size(); ++i) {
    auto const& t = terminals[i];
    spherical_point t_location(t.lng_, t.lat_);

    std::vector<value> results;
    rtree.query(bgi::intersects(generate_box(t_location, MAX_BIKE_DIST)) &&
                    bgi::satisfies([&t_location](const value& v) {
                      return distance_in_m(v.first, t_location) < MAX_BIKE_DIST;
                    }),
                std::back_inserter(results));

    std::vector<close_location> close_terminals;
    for (const auto& result : results) {
      int dist = distance_in_m(result.first, t_location);
      if (dist < 500) {
        continue;
      }
      int const dur = dist * LINEAR_DIST_APPROX / BIKE_SPEED;
      close_terminals.push_back({terminals[result.second].uid_, dur});
    }
    all_close_terminals.push_back(close_terminals);
  }

  return all_close_terminals;
}

}  // namespace motis::bikesharing
