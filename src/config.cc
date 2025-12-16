#include "motis/config.h"

#include <iostream>

#include "boost/url.hpp"

#include "fmt/std.h"

#include "utl/erase.h"
#include "utl/overloaded.h"
#include "utl/read_file.h"
#include "utl/verify.h"

#include "nigiri/clasz.h"
#include "nigiri/routing/limits.h"

#include "rfl.hpp"
#include "rfl/yaml.hpp"

namespace fs = std::filesystem;

namespace motis {

template <rfl::internal::StringLiteral Name>
consteval auto drop_last() {
  return []<size_t... Is>(std::index_sequence<Is...>) {
    return rfl::internal::StringLiteral<Name.arr_.size() - 1>(Name.arr_[Is]...);
  }(std::make_index_sequence<Name.arr_.size() - 2>{});
}

struct drop_trailing {
public:
  template <typename StructType>
  static auto process(auto&& named_tuple) {
    auto const handle_one = []<typename FieldType>(FieldType&& f) {
      if constexpr (FieldType::name() != "xml_content" &&
                    !rfl::internal::is_rename_v<typename FieldType::Type>) {
        return handle_one_field(std::move(f));
      } else {
        return std::move(f);
      }
    };
    return named_tuple.transform(handle_one);
  }

private:
  template <typename FieldType>
  static auto handle_one_field(FieldType&& _f) {
    using NewFieldType =
        rfl::Field<drop_last<FieldType::name_>(), typename FieldType::Type>;
    return NewFieldType(_f.value());
  }
};

std::ostream& operator<<(std::ostream& out, config const& c) {
  return out << rfl::yaml::write<drop_trailing>(c);
}

config config::read_simple(std::vector<std::string> const& args) {
  auto c = config{};
  for (auto const& arg : args) {
    auto const p = fs::path{arg};
    utl::verify(fs::exists(p), "path {} does not exist", p);
    if (fs::is_regular_file(p) && p.generic_string().ends_with("osm.pbf")) {
      c.osm_ = p;
      c.street_routing_ = true;
      c.geocoding_ = true;
      c.reverse_geocoding_ = true;
      c.tiles_ = {config::tiles{.profile_ = "tiles-profiles/full.lua"}};
    } else {
      if (!c.timetable_.has_value()) {
        c.timetable_ = {timetable{.railviz_ = true}};
      }

      auto tag = p.stem().generic_string();
      utl::erase(tag, '_');
      utl::erase(tag, '.');
      c.timetable_->datasets_.emplace(
          tag, timetable::dataset{.path_ = p.generic_string()});
    }
  }
  return c;
}

config config::read(std::filesystem::path const& p) {
  auto const file_content = utl::read_file(p.generic_string().c_str());
  utl::verify(file_content.has_value(), "could not read config file at {}", p);
  return read(*file_content);
}

config config::read(std::string const& s) {
  auto c =
      rfl::yaml::read<config, drop_trailing, rfl::DefaultIfMissing>(s).value();
  if (!c.limits_.has_value()) {
    c.limits_.emplace(limits{});
  }
  c.verify();
  return c;
}

void config::verify() const {
  auto const street_routing = use_street_routing();

  utl::verify(!tiles_ || osm_, "feature TILES requires OpenStreetMap data");
  utl::verify(!street_routing || osm_,
              "feature STREET_ROUTING requires OpenStreetMap data");
  utl::verify(!timetable_ || !timetable_->datasets_.empty(),
              "feature TIMETABLE requires timetable data");
  utl::verify(
      !osr_footpath_ || (street_routing && timetable_),
      "feature OSR_FOOTPATH requires features STREET_ROUTING and TIMETABLE");
  utl::verify(!has_elevators() || (street_routing && timetable_),
              "feature ELEVATORS requires STREET_ROUTING and TIMETABLE");
  utl::verify(!has_gbfs_feeds() || street_routing,
              "feature GBFS requires feature STREET_ROUTING");
  utl::verify(!has_prima() || (street_routing && timetable_),
              "feature ODM requires feature STREET_ROUTING");
  utl::verify(!has_elevators() || osr_footpath_,
              "feature ELEVATORS requires feature OSR_FOOTPATHS");
  utl::verify(limits_.value().plan_max_search_window_minutes_ <=
                  nigiri::routing::kMaxSearchIntervalSize.count(),
              "plan_max_search_window_minutes limit cannot be above {}",
              nigiri::routing::kMaxSearchIntervalSize.count());

  if (timetable_) {
    for (auto const& [id, d] : timetable_->datasets_) {
      utl::verify(!id.contains("_"), "dataset identifier may not contain '_'");
      if (d.rt_.has_value()) {
        for (auto const& rt : *d.rt_) {
          try {
            boost::urls::url{rt.url_};
          } catch (std::exception const& e) {
            throw utl::fail("{} is not a valid url: {}", rt.url_, e.what());
          }
          utl::verify(rt.protocol_ != timetable::dataset::rt::protocol::auser ||
                          timetable_->incremental_rt_update_,
                      "VDV AUS requires incremental RT update scheme");
        }
      }
    }
  }
}

void config::verify_input_files_exist() const {
  utl::verify(!osm_ || fs::is_regular_file(*osm_),
              "OpenStreetMap file does not exist: {}",
              osm_.value_or(fs::path{}));

  utl::verify(!tiles_ || fs::is_regular_file(tiles_->profile_),
              "tiles profile {} does not exist",
              tiles_.value_or(tiles{}).profile_);

  utl::verify(!tiles_ || !tiles_->coastline_ ||
                  fs::is_regular_file(*tiles_->coastline_),
              "coastline file {} does not exist",
              tiles_.value_or(tiles{}).coastline_.value_or(""));

  if (timetable_) {
    for (auto const& [tag, d] : timetable_->datasets_) {
      utl::verify(d.path_.starts_with("\n#") || fs::is_directory(d.path_) ||
                      fs::is_regular_file(d.path_),
                  "timetable dataset {} does not exist: {}", tag, d.path_);

      utl::verify(!d.script_.has_value() ||
                      d.script_->starts_with("\nfunction") ||
                      fs::is_regular_file(*d.script_),
                  "user script for {} not found at path: \"{}\"", tag,
                  d.script_.value_or(""));

      if (d.clasz_bikes_allowed_.has_value()) {
        for (auto const& c : *d.clasz_bikes_allowed_) {
          nigiri::to_clasz(c.first);
        }
      }
    }
  }
}

bool config::requires_rt_timetable_updates() const {
  return timetable_.has_value() &&
         ((has_elevators() && get_elevators()->url_.has_value()) ||
          utl::any_of(timetable_->datasets_, [](auto&& d) {
            return d.second.rt_.has_value() && !d.second.rt_->empty();
          }));
}

bool config::has_gbfs_feeds() const {
  return gbfs_.has_value() && !gbfs_->feeds_.empty();
}

bool config::has_prima() const { return prima_.has_value(); }

unsigned config::n_threads() const {
  return server_
      .and_then([](config::server const& s) {
        return s.n_threads_ == 0U ? std::nullopt : std::optional{s.n_threads_};
      })
      .value_or(std::thread::hardware_concurrency());
}

std::optional<config::elevators> const& config::get_elevators() const {
  utl::verify(has_elevators(),
              "config::get_elevators() requires config::has_elevators()");
  return std::get<std::optional<elevators>>(elevators_);
}

bool config::has_elevators() const {
  return std::visit(
      utl::overloaded{
          [](std::optional<elevators> const& x) { return x.has_value(); },
          [](bool const x) {
            utl::verify(!x, "elevators=true is not supported");
            return x;
          }},
      elevators_);
}

std::optional<config::street_routing> config::get_street_routing() const {
  return std::visit(
      utl::overloaded{
          [](std::optional<config::street_routing> const& x) { return x; },
          [](bool const street_routing) {
            return street_routing ? std::optional{config::street_routing{}}
                                  : std::nullopt;
          }},
      street_routing_);
}

bool config::use_street_routing() const {
  return std::visit(
      utl::overloaded{
          [](std::optional<street_routing> const& o) { return o.has_value(); },
          [](bool const b) { return b; },
      },
      street_routing_);
}

}  // namespace motis
