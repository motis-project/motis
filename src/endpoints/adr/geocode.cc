#include "motis/endpoints/adr/geocode.h"

#include "boost/thread/tss.hpp"

#include "utl/for_each_bit_set.h"
#include "utl/to_vec.h"

#include "fmt/format.h"

#include "nigiri/timetable.h"

#include "adr/adr.h"
#include "adr/typeahead.h"

#include "motis/endpoints/adr/suggestions_to_response.h"

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

  auto& ctx = get_guess_context(t_, cache_);

  auto lang_indices = std::basic_string<a::language_idx_t>{{a::kDefaultLang}};
  if (params.language_.has_value()) {
    auto const l_idx = t_.resolve_language(*params.language_);
    if (l_idx != a::language_idx_t::invalid()) {
      lang_indices.push_back(l_idx);
    }
  }

  auto const token_pos = a::get_suggestions<false>(
      t_, geo::latlng{0, 0}, params.text_, 10U, lang_indices, ctx);

  return suggestions_to_response(t_, tt_, lang_indices, token_pos,
                                 ctx.suggestions_);
}

}  // namespace motis::ep