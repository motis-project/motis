#include "motis/address/address.h"

#include <fstream>
#include <istream>
#include <regex>
#include <sstream>

#include "boost/filesystem.hpp"

#include "cereal/archives/binary.hpp"

#include "utl/to_vec.h"

#include "address-typeahead/common.h"
#include "address-typeahead/extract.h"
#include "address-typeahead/serialization.h"
#include "address-typeahead/typeahead.h"

#include "motis/core/common/logging.h"
#include "motis/module/event_collector.h"
#include "motis/module/ini_io.h"

using namespace motis::module;
using namespace address_typeahead;

namespace motis::address {

struct import_state {
  CISTA_COMPARABLE()
  named<std::string, MOTIS_NAME("path")> path_;
  named<cista::hash_t, MOTIS_NAME("hash")> hash_;
  named<size_t, MOTIS_NAME("size")> size_;
};

struct address::impl {
  explicit impl(std::string const& path) {
    auto in = std::ifstream(path.c_str(), std::ios::binary);
    in.exceptions(std::ios_base::failbit);

    cereal::BinaryInputArchive ia(in);
    ia(context_);
    t_ = std::make_unique<address_typeahead::typeahead>(context_);
  }

  std::pair<std::string, std::vector<address_typeahead::index_t>> guess(
      std::string const& str) const {
    auto ss = std::stringstream{str};
    auto sub_strings = std::vector<std::string>{};
    auto buf = std::string{};

    while (ss >> buf) {
      sub_strings.emplace_back(buf);
    }

    auto house_number = std::string{};
    for (auto str_it = begin(sub_strings); str_it != end(sub_strings);) {
      if (std::regex_match(*str_it, std::regex("\\.|\\d{1,4}[:alpha:]*"))) {
        house_number = *str_it;
        str_it = sub_strings.erase(str_it);
      } else {
        ++str_it;
      }
    }

    if (sub_strings.empty()) {
      return {"", {}};
    }

    address_typeahead::complete_options options;
    options.max_results_ = 10;
    options.string_chain_len_ = 2;
    return {house_number, t_->complete(sub_strings, options)};
  }

  msg_ptr get_guesses(msg_ptr const& msg) const {
    std::string house_number;
    std::vector<address_typeahead::index_t> guess_indices;
    std::tie(house_number, guess_indices) =
        guess(motis_content(AddressRequest, msg)->input()->str());

    message_creator fbb;
    fbb.create_and_finish(
        MsgContent_AddressResponse,
        CreateAddressResponse(
            fbb,
            fbb.CreateVector(utl::to_vec(
                guess_indices,
                [&](address_typeahead::index_t id) {
                  double lat = 0, lng = 0;
                  if (!house_number.empty() &&
                      context_.coordinates_for_house_number(id, house_number,
                                                            lat, lng)) {
                  } else {
                    context_.get_coordinates(id, lat, lng);
                  }
                  auto const pos = Position{lat, lng};

                  return CreateAddress(
                      fbb, &pos, fbb.CreateString(context_.get_name(id)),
                      fbb.CreateString(context_.is_place(id)
                                           ? "place"
                                           : context_.is_street(id) ? "street"
                                                                    : "unkown"),
                      fbb.CreateVector(utl::to_vec(
                          context_.get_area_names(id),
                          [&](std::pair<std::string, uint32_t> const& region) {
                            return CreateRegion(fbb,
                                                fbb.CreateString(region.first),
                                                region.second);
                          })));
                })))
            .Union());
    return make_msg(fbb);
  }

  address_typeahead::typeahead_context context_;
  std::unique_ptr<address_typeahead::typeahead> t_;
};

address::address() : module("Address Typeahead", "address") {}

address::~address() = default;

std::string address::db_file() const {
  return (get_data_directory() / "address" / "address_db.raw").generic_string();
}

void address::import(motis::module::registry& reg) {
  std::make_shared<event_collector>(
      get_data_directory().generic_string(), "address", reg,
      [this](std::map<std::string, msg_ptr> const& dependencies) {
        using import::OSMEvent;

        auto const dir = get_data_directory() / "address";
        auto const osm = motis_content(OSMEvent, dependencies.at("OSM"));
        auto const state = import_state{data_path(osm->path()->str()),
                                        osm->hash(), osm->size()};

        if (read_ini<import_state>(dir / "import.ini") != state) {
          boost::filesystem::create_directories(dir);
          std::ofstream out{db_file().c_str(), std::ios::binary};
          address_typeahead::extract(osm->path()->str(), out);
          write_ini(dir / "import.ini", state);
        }

        import_successful_ = true;
      })
      ->require("OSM", [](msg_ptr const& msg) {
        return msg->get()->content_type() == MsgContent_OSMEvent;
      });
}

void address::init(motis::module::registry& reg) {
  auto in = std::ifstream(db_file(), std::ios::binary);
  in.exceptions(std::ios_base::failbit);

  address_typeahead::typeahead_context context;
  {
    cereal::BinaryInputArchive ia{in};
    ia(context);
  }

  address_typeahead::typeahead t{context};

  impl_ = std::make_unique<impl>(db_file());
  reg.register_op("/address", [this](msg_ptr const& msg) {
    return impl_->get_guesses(msg);
  });
}

}  // namespace motis::address
