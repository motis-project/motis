#include "motis/address/address.h"

#include <fstream>
#include <istream>
#include <regex>
#include <sstream>

#include "cereal/archives/binary.hpp"

#include "utl/to_vec.h"

#include "address-typeahead/common.h"
#include "address-typeahead/serialization.h"
#include "address-typeahead/typeahead.h"

#include "motis/core/common/logging.h"

using namespace motis::module;
using namespace address_typeahead;

namespace motis::address {

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

address::address() : module("Address Typeahead", "address") {
  param(db_path_, "db", "address typeahead database path");
}

address::~address() = default;

void address::init(motis::module::registry& reg) {
  try {
    auto in = std::ifstream(db_path_, std::ios::binary);
    in.exceptions(std::ios_base::failbit);

    address_typeahead::typeahead_context context;
    {
      cereal::BinaryInputArchive ia(in);
      ia(context);
    }

    address_typeahead::typeahead t(context);

    impl_ = std::make_unique<impl>(db_path_);
    reg.register_op("/address", [this](msg_ptr const& msg) {
      return impl_->get_guesses(msg);
    });
  } catch (std::exception const& e) {
    LOG(logging::warn) << "address module not initialized (" << e.what() << ")";
  }
}

}  // namespace motis::address
