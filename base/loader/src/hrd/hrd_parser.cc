#include "motis/loader/hrd/hrd_parser.h"

#include "utl/enumerate.h"
#include "utl/erase.h"

#include "motis/core/common/logging.h"

#include "motis/schedule-format/Schedule_generated.h"

#include "motis/loader/hrd/builder/attribute_builder.h"
#include "motis/loader/hrd/builder/bitfield_builder.h"
#include "motis/loader/hrd/builder/category_builder.h"
#include "motis/loader/hrd/builder/direction_builder.h"
#include "motis/loader/hrd/builder/footpath_builder.h"
#include "motis/loader/hrd/builder/line_builder.h"
#include "motis/loader/hrd/builder/meta_station_builder.h"
#include "motis/loader/hrd/builder/provider_builder.h"
#include "motis/loader/hrd/builder/route_builder.h"
#include "motis/loader/hrd/builder/rule_service_builder.h"
#include "motis/loader/hrd/builder/service_builder.h"
#include "motis/loader/hrd/builder/station_builder.h"
#include "motis/loader/hrd/parser/attributes_parser.h"
#include "motis/loader/hrd/parser/basic_info_parser.h"
#include "motis/loader/hrd/parser/bitfields_parser.h"
#include "motis/loader/hrd/parser/categories_parser.h"
#include "motis/loader/hrd/parser/directions_parser.h"
#include "motis/loader/hrd/parser/merge_split_rules_parser.h"
#include "motis/loader/hrd/parser/providers_parser.h"
#include "motis/loader/hrd/parser/service_parser.h"
#include "motis/loader/hrd/parser/stations_parser.h"
#include "motis/loader/hrd/parser/through_services_parser.h"
#include "motis/loader/hrd/parser/timezones_parser.h"
#include "motis/loader/hrd/parser/track_rules_parser.h"
#include "motis/loader/parser_error.h"
#include "motis/loader/util.h"

namespace motis::loader::hrd {

using namespace flatbuffers64;
using namespace utl;
using namespace motis::logging;
namespace fs = boost::filesystem;

bool hrd_parser::applicable(fs::path const& path) {
  return std::any_of(begin(configs), end(configs),
                     [&](const config& c) { return applicable(path, c); });
}

bool hrd_parser::applicable(fs::path const& path, config const& c) {
  auto const core_data_root = path / c.core_data_;
  return std::all_of(
      begin(c.required_files_), end(c.required_files_),
      [&core_data_root](std::vector<std::string> const& alternatives) {
        if (alternatives.empty()) {
          return true;
        }
        return std::any_of(
            begin(alternatives), end(alternatives),
            [&core_data_root](std::string const& filename) {
              return filename.empty() /* file not required */ ||
                     fs::is_regular_file(core_data_root / filename);
            });
      });
}

std::vector<std::string> hrd_parser::missing_files(
    fs::path const& hrd_root) const {
  for (auto const& c : configs) {
    if (fs::is_regular_file(hrd_root / c.core_data_ / c.files(BASIC_DATA))) {
      return missing_files(hrd_root, c);
    }
  }
  return {"eckdaten.*"};
}

std::vector<std::string> hrd_parser::missing_files(fs::path const& hrd_root,
                                                   config const& c) {
  std::vector<std::string> missing_files;
  auto const schedule_data_root = hrd_root / c.fplan_;
  if (!fs::exists(schedule_data_root)) {
    missing_files.push_back(schedule_data_root.string());
  }

  auto const core_data_root = hrd_root / c.core_data_;
  for (auto const& alternatives : c.required_files_) {
    std::vector<int> missing_indices;
    int pos = 0;
    for (auto const& alternative : alternatives) {
      if (!fs::is_regular_file(core_data_root / alternative)) {
        missing_indices.push_back(pos);
      }
      ++pos;
    }
    if (missing_indices.size() < alternatives.size()) {
      continue;
    }
    for (auto const idx : missing_indices) {
      missing_files.emplace_back(
          (core_data_root / alternatives[idx]).generic_string());
    }
  }

  return missing_files;
}

loaded_file load(fs::path const& root, filename_key k,
                 std::vector<std::vector<std::string>> const& required_files) {
  if (required_files[k].empty()) {  // not available for this HRD version.
    return loaded_file{};
  }

  auto it = std::find_if(begin(required_files[k]), end(required_files[k]),
                         [&root](std::string const& filename) {
                           return fs::is_regular_file(root / filename);
                         });
  utl::verify(it != end(required_files[k]),
              "unable to load non-regular file(s): filename_key={}",
              static_cast<int>(k));
  return loaded_file(root / *it);
}

void parse_and_build_services(
    fs::path const& hrd_root, std::map<int, bitfield> const& bitfields,
    std::vector<std::unique_ptr<loaded_file>>& schedule_data,
    std::function<void(hrd_service const&)> const& service_builder_fun,
    config const& c) {
  std::vector<fs::path> files;
  collect_files(hrd_root / c.fplan_, files);
  for (auto const& [i, file] : utl::enumerate(files)) {
    schedule_data.emplace_back(std::make_unique<loaded_file>(file));
    LOG(info) << "parsing " << i << "/" << files.size() << " "
              << schedule_data.back()->name();
    for_each_service(*schedule_data.back(), bitfields, service_builder_fun, c);
  }
}

void delete_f_equivalences(
    std::set<station_meta_data::footpath> const& hrd_footpaths,
    std::set<station_meta_data::meta_station>& hrd_meta_stations) {
  for (auto const& f : hrd_footpaths) {
    if (f.f_equal_) {
      auto meta = hrd_meta_stations.find({f.from_eva_num_, {}});
      if (meta == hrd_meta_stations.end()) {
        continue;
      }
      auto meta_copy = *meta;
      utl::erase(meta_copy.equivalent_, f.to_eva_num_);
      hrd_meta_stations.erase({f.from_eva_num_, {}});
      if (!meta_copy.equivalent_.empty()) {
        hrd_meta_stations.emplace(meta_copy);
      }
    }
  }
}

void hrd_parser::parse(fs::path const& hrd_root, FlatBufferBuilder& fbb) {
  for (auto const& c : configs) {
    if (fs::is_regular_file(hrd_root / c.core_data_ / c.files(BASIC_DATA))) {
      return parse(hrd_root, fbb, c);
    } else {
      LOG(info) << (hrd_root / c.core_data_ / c.files(BASIC_DATA))
                << " does not exist";
    }
  }
  throw std::runtime_error{"no parser was applicable"};
}

void hrd_parser::parse(fs::path const& hrd_root, FlatBufferBuilder& fbb,
                       config const& c) {
  LOG(info) << "parsing HRD data version " << c.version_;

  auto const core_data_root = hrd_root / c.core_data_;
  auto const bitfields_file =
      load(core_data_root, BITFIELDS, c.required_files_);
  bitfield_builder bb(parse_bitfields(bitfields_file, c));
  auto const infotext_file = load(core_data_root, INFOTEXT, c.required_files_);
  auto const stations_file = load(core_data_root, STATIONS, c.required_files_);
  auto const coordinates_file =
      load(core_data_root, COORDINATES, c.required_files_);
  auto const timezones_file =
      load(core_data_root, TIMEZONES, c.required_files_);
  auto const basic_data_file =
      load(core_data_root, BASIC_DATA, c.required_files_);
  auto const footp_file_1 = load(core_data_root, FOOTPATHS, c.required_files_);
  auto const footp_file_ext =
      load(core_data_root, FOOTPATHS_EXT, c.required_files_);
  auto const minct_file = load(core_data_root, MIN_CT_FILE, c.required_files_);
  station_meta_data metas;
  parse_station_meta_data(infotext_file, footp_file_1, footp_file_ext,
                          minct_file, metas, c);

  station_builder stb(parse_stations(stations_file, coordinates_file, metas, c),
                      parse_timezones(timezones_file, basic_data_file, c));

  auto const categories_file =
      load(core_data_root, CATEGORIES, c.required_files_);
  category_builder cb(parse_categories(categories_file, c));

  auto const providers_file =
      load(core_data_root, PROVIDERS, c.required_files_);
  provider_builder pb(parse_providers(providers_file, c));

  auto const attributes_file =
      load(core_data_root, ATTRIBUTES, c.required_files_);
  attribute_builder ab(parse_attributes(attributes_file, c));

  auto const directions_file =
      load(core_data_root, DIRECTIONS, c.required_files_);
  direction_builder db(parse_directions(directions_file, c));

  auto const tracks_file = load(core_data_root, TRACKS, c.required_files_);
  service_builder sb(parse_track_rules(tracks_file, fbb, c));

  line_builder lb;
  route_builder rb;

  service_rules rules;
  auto const ts_file =
      load(core_data_root, THROUGH_SERVICES, c.required_files_);
  parse_through_service_rules(ts_file, bb.hrd_bitfields_, rules, c);

  auto const ms_file =
      load(core_data_root, MERGE_SPLIT_SERVICES, c.required_files_);
  parse_merge_split_service_rules(ms_file, bb.hrd_bitfields_, rules, c);

  rule_service_builder rsb(rules);

  // parse and build services
  std::vector<std::unique_ptr<loaded_file>> schedule_data;
  parse_and_build_services(
      hrd_root, bb.hrd_bitfields_, schedule_data,
      [&](hrd_service const& s) {
        if (!rsb.add_service(s)) {
          sb.create_service(s, rb, stb, cb, pb, lb, ab, bb, db, fbb, false);
        }
      },
      c);

  // compute and build rule services
  rsb.resolve_rule_services();
  rsb.create_rule_services(
      [&](hrd_service const& s, bool is_rule_service, FlatBufferBuilder& fbb) {
        return sb.create_service(s, rb, stb, cb, pb, lb, ab, bb, db, fbb,
                                 is_rule_service);
      },
      stb, fbb);

  if (c.version_ == "hrd_5_00_8") {
    delete_f_equivalences(metas.footpaths_, metas.meta_stations_);
  }

  auto interval = parse_interval(basic_data_file);
  auto schedule_name = parse_schedule_name(basic_data_file);
  auto footpaths = create_footpaths(metas.footpaths_, stb, fbb);
  auto metastations = create_meta_stations(metas.meta_stations_, stb, fbb);
  fbb.Finish(CreateSchedule(fbb,
                            fbb.CreateVectorOfSortedTables(&sb.fbs_services_),
                            fbb.CreateVector(values(stb.fbs_stations_)),
                            fbb.CreateVector(values(rb.routes_)), &interval,
                            footpaths, fbb.CreateVector(rsb.fbs_rule_services_),
                            metastations, fbb.CreateString(schedule_name)));
}

}  // namespace motis::loader::hrd
