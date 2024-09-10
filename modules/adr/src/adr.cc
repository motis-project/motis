#include "motis/adr/adr.h"

#include <filesystem>
#include <fstream>
#include <istream>
#include <ranges>
#include <regex>
#include <sstream>

#include "boost/thread/tss.hpp"

#include "opentelemetry/trace/scope.h"
#include "opentelemetry/trace/span.h"

#include "cista/reflection/comparable.h"

#include "utl/to_vec.h"

#include "adr/adr.h"
#include "adr/typeahead.h"

#include "motis/core/common/logging.h"
#include "motis/core/conv/position_conv.h"
#include "motis/core/otel/tracer.h"
#include "motis/module/event_collector.h"
#include "motis/module/ini_io.h"

namespace fs = std::filesystem;
namespace mm = motis::module;
namespace a = adr;

namespace motis::adr {

struct import_state {
  CISTA_COMPARABLE()
  mm::named<std::string, MOTIS_NAME("path")> path_;
  mm::named<cista::hash_t, MOTIS_NAME("hash")> hash_;
  mm::named<size_t, MOTIS_NAME("size")> size_;
};

struct adr::impl {
  explicit impl(cista::wrapped<a::typeahead> t) : t_{std::move(t)} {}

  a::guess_context& get_guess_context() {
    auto static ctx = boost::thread_specific_ptr<a::guess_context>{};
    if (ctx.get() == nullptr) {
      ctx.reset(new a::guess_context{cache_});
    }
    ctx->resize(*t_);
    return *ctx;
  }

  cista::wrapped<a::typeahead> t_;
  a::cache cache_{.n_strings_ = t_->strings_.size(), .max_size_ = 100U};
};

adr::adr() : module("Address Typeahead", "adr") {}

adr::~adr() = default;

void adr::import(motis::module::import_dispatcher& reg) {
  std::make_shared<mm::event_collector>(
      get_data_directory().generic_string(), "adr", reg,
      [this](mm::event_collector::dependencies_map_t const& dependencies,
             mm::event_collector::publish_fn_t const&) {
        using import::OSMEvent;

        auto span = motis_tracer->StartSpan("adr::import");
        auto scope = opentelemetry::trace::Scope{span};

        auto const dir = get_data_directory() / "adr";
        auto const osm = motis_content(OSMEvent, dependencies.at("OSM"));
        auto const state = import_state{data_path(osm->path()->str()),
                                        osm->hash(), osm->size()};

        span->SetAttribute("motis.osm.file", osm->path()->str());
        span->SetAttribute("motis.osm.size", osm->size());

        if (mm::read_ini<import_state>(dir / "import.ini") != state) {
          span->SetAttribute("motis.import.state", "changed");
          fs::create_directories(dir);
          a::extract(osm->path()->str(), (dir / "adr"), dir);
          mm::write_ini(dir / "import.ini", state);
        } else {
          span->SetAttribute("motis.import.state", "unchanged");
        }

        impl_ = std::make_unique<impl>(a::read(dir / "adr.t.adr", false));
        import_successful_ = true;
      })
      ->require("OSM", [](mm::msg_ptr const& msg) {
        return msg->get()->content_type() == MsgContent_OSMEvent;
      });
}

mm::msg_ptr adr::guess(mm::msg_ptr const& msg) {
  using motis::address::AddressRequest;
  auto const req = motis_content(AddressRequest, msg);

  auto lang_indices = std::basic_string<a::language_idx_t>{{a::kDefaultLang}};
  auto& ctx = impl_->get_guess_context();
  a::get_suggestions<false>(*impl_->t_, geo::latlng{0, 0}, req->input()->view(),
                            10U, lang_indices, ctx);

  mm::message_creator fbb;
  fbb.create_and_finish(
      MsgContent_AddressResponse,
      CreateAddressResponse(
          fbb,
          fbb.CreateVector(utl::to_vec(
              ctx.suggestions_,
              [&](a::suggestion const& s) {
                auto const pos = to_fbs(s.coordinates_.as_latlng());
                auto const regions = fbb.CreateVector(utl::to_vec(
                    impl_->t_->area_sets_[s.area_set_] | std::views::reverse,
                    [&](a::area_idx_t const area_idx) {
                      auto const admin_lvl =
                          impl_->t_->area_admin_level_[area_idx];
                      auto const name =
                          impl_->t_
                              ->strings_[impl_->t_->area_names_
                                             [area_idx][a::kDefaultLangIdx]]
                              .view();
                      return address::CreateRegion(fbb, fbb.CreateString(name),
                                                   to_idx(admin_lvl));
                    }));

                return std::visit(
                    utl::overloaded{
                        [&](a::place_idx_t) {
                          return address::CreateAddress(
                              fbb, &pos,
                              fbb.CreateString(
                                  impl_->t_->strings_[s.str_].view()),
                              fbb.CreateString("place"), regions);
                        },
                        [&](a::address const addr) {
                          auto const is_address =
                              addr.house_number_ != a::address::kNoHouseNumber;
                          auto const name =
                              is_address
                                  ? fmt::format(
                                        "{} {}",
                                        impl_->t_->strings_[s.str_].view(),
                                        impl_->t_
                                            ->strings_[impl_->t_->house_numbers_
                                                           [addr.street_]
                                                           [addr.house_number_]]
                                            .view())
                                  : fmt::format(
                                        "{}",
                                        impl_->t_->strings_[s.str_].view());
                          return address::CreateAddress(
                              fbb, &pos, fbb.CreateString(name),
                              fbb.CreateString(is_address ? "address"
                                                          : "street"),
                              regions);
                        }},
                    s.location_);
              })))
          .Union());
  return make_msg(fbb);
}

void adr::init(mm::registry& reg) {
  reg.register_op("/address",
                  [&](mm::msg_ptr const& msg) { return this->guess(msg); }, {});
}

}  // namespace motis::adr
