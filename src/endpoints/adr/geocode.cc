#include "motis/endpoints/adr/geocode.h"

#include "boost/thread/tss.hpp"

#include "utl/for_each_bit_set.h"
#include "utl/to_vec.h"

#include "fmt/format.h"

#include "nigiri/timetable.h"

#include "adr/adr.h"
#include "adr/typeahead.h"

#include "motis/endpoints/adr/filter_conv.h"
#include "motis/endpoints/adr/suggestions_to_response.h"
#include "motis/parse_location.h"

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
    utl::verify(parsed.has_value(), "could not parse place {}", s);
    return std::optional{parsed.value().pos_};
  });

  auto& ctx = get_guess_context(t_, cache_);

  auto lang_indices = basic_string<a::language_idx_t>{{a::kDefaultLang}};
  if (params.language_.has_value()) {
    auto const l_idx = t_.resolve_language(*params.language_);
    if (l_idx != a::language_idx_t::invalid()) {
      lang_indices.push_back(l_idx);
    }
  }
  auto const token_pos = a::get_suggestions<false>(
      t_, params.text_, 10U, lang_indices, ctx, place,
      static_cast<float>(params.placeBias_), to_filter_type(params.type_));
  return suggestions_to_response(t_, tt_, tags_, w_, pl_, matches_,
                                 lang_indices, token_pos, ctx.suggestions_);
}

}  // namespace motis::ep