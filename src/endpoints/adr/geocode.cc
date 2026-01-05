#include "motis/endpoints/adr/geocode.h"

#include "boost/thread/tss.hpp"

#include "utl/for_each_bit_set.h"
#include "utl/to_vec.h"

#include "fmt/format.h"

#include "net/bad_request_exception.h"

#include "nigiri/timetable.h"

#include "adr/adr.h"
#include "adr/typeahead.h"

#include "motis/endpoints/adr/filter_conv.h"
#include "motis/endpoints/adr/suggestions_to_response.h"
#include "motis/parse_location.h"
#include "motis/timetable/modes_to_clasz_mask.h"

namespace n = nigiri;
namespace a = adr;

namespace motis::ep {

a::guess_context& get_guess_context(a::typeahead const& t, a::cache& cache) {
  auto static ctx = boost::thread_specific_ptr<a::guess_context>{};
  if (ctx.get() == nullptr) {
    ctx.reset(new a::guess_context{cache});
  }
  ctx->resize(t);
  return *ctx;
}

api::geocode_response geocode::operator()(
    boost::urls::url_view const& url) const {
  auto const params = api::geocode_params{url.params()};
  auto const place = params.place_.and_then([](std::string const& s) {
    auto const parsed = parse_location(s);
    utl::verify<net::bad_request_exception>(parsed.has_value(),
                                            "could not parse place {}", s);
    return std::optional{parsed.value().pos_};
  });
  auto const allowed_modes =
      params.mode_.transform([](std::vector<api::ModeEnum> const& modes) {
        return to_clasz_mask(modes);
      });

  auto& ctx = get_guess_context(t_, cache_);

  auto lang_indices = basic_string<a::language_idx_t>{{a::kDefaultLang}};
  if (params.language_.has_value()) {
    for (auto const& language : *params.language_) {
      auto const l_idx = t_.resolve_language(language);
      if (l_idx != a::language_idx_t::invalid()) {
        lang_indices.push_back(l_idx);
      }
    }
  }
  auto const place_filter =
      allowed_modes
          .transform([&](n::routing::clasz_mask_t allowed) {
            return std::function{[allowed, this](adr::place_idx_t place_idx) {
              if (t_.place_type_[place_idx] != adr::amenity_category::kExtra) {
                return true;
              }
              auto const i = adr_extra_place_idx_t{
                  static_cast<adr_extra_place_idx_t::value_t>(place_idx -
                                                              t_.ext_start_)};
              auto const available = ae_->place_clasz_[i];
              return static_cast<bool>(available & allowed);
            }};
          })
          .value_or(std::function<bool(adr::place_idx_t)>{});
  auto const token_pos =
      a::get_suggestions<false>(t_, params.text_, 10U, lang_indices, ctx, place,
                                static_cast<float>(params.placeBias_),
                                to_filter_type(params.type_), place_filter);
  return suggestions_to_response(t_, f_, ae_, tt_, tags_, w_, pl_, matches_,
                                 lang_indices, token_pos, ctx.suggestions_);
}

}  // namespace motis::ep
