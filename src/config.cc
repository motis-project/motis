#include "motis/config.h"

#include <iostream>

#include "fmt/std.h"

#include "utl/read_file.h"
#include "utl/verify.h"

#include "nigiri/clasz.h"

#include "rfl.hpp"
#include "rfl/yaml.hpp"

namespace fs = std::filesystem;

namespace motis {

template <rfl::internal::StringLiteral Name, size_t I = 0, char... Chars>
consteval auto drop_last() {
  if constexpr (I == Name.arr_.size() - 2) {
    return rfl::internal::StringLiteral<sizeof...(Chars) + 1>(Chars...);
  } else {
    return drop_last<Name, I + 1, Chars..., Name.arr_[I]>();
  }
}

struct drop_trailing {
public:
  template <typename StructType>
  static auto process(auto&& named_tuple) {
    const auto handle_one = []<typename FieldType>(FieldType&& f) {
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

config config::read(std::filesystem::path const& p) {
  auto const file_content = utl::read_file(p.generic_string().c_str());
  utl::verify(file_content.has_value(), "could not read config file at {}", p);
  return read(*file_content);
}

config config::read(std::string const& s) {
  auto c =
      rfl::yaml::read<config, drop_trailing, rfl::DefaultIfMissing>(s).value();
  c.verify();
  return c;
}

void config::verify() const {
  utl::verify(!geocoding_ || osm_,
              "feature GEOCODING requires OpenStreetMap data");
  utl::verify(!reverse_geocoding_ || (geocoding_ && osm_),
              "feature REVERSE_GEOCODING requires OpenStreetMap data and "
              "feature GEOCODING");
  utl::verify(!tiles_ || osm_, "feature TILES requires OpenStreetMap data");
  utl::verify(!street_routing_ || osm_,
              "feature STREET_ROUTING requires OpenStreetMap data");
  utl::verify(!timetable_ || !timetable_->datasets_.empty(),
              "feature TIMETABLE requires timetable data");
  utl::verify(
      !osr_footpath_ || (street_routing_ && timetable_),
      "feature OSR_FOOTPATH requires features STREET_ROUTING and TIMETABLE");
  utl::verify(
      !elevators_ || (fasta_ && street_routing_ && timetable_),
      "feature ELEVATORS requires fasta.json and features STREET_ROUTING and "
      "TIMETABLE");
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
    for (auto const& [_, d] : timetable_->datasets_) {
      utl::verify(d.path_.starts_with("\n#") || fs::is_directory(d.path_) ||
                      fs::is_regular_file(d.path_),
                  "timetable dataset does not exist: {}", d.path_);

      if (d.clasz_bikes_allowed_) {
        for (auto const& c : *d.clasz_bikes_allowed_) {
          nigiri::to_clasz(c.first);
        }
      }
    }
  }
}

bool config::requires_rt_timetable_updates() const {
  return timetable_.has_value() &&
         utl::any_of(timetable_->datasets_, [](auto&& d) {
           return d.second.rt_.has_value() && !d.second.rt_->empty();
         });
}

}  // namespace motis