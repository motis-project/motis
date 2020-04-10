#include "motis/loader/hrd/builder/provider_builder.h"

#include "utl/get_or_create.h"

#include "motis/loader/hrd/parse_config.h"
#include "motis/loader/util.h"

namespace motis::loader::hrd {

using namespace utl;
using namespace flatbuffers64;

provider_builder::provider_builder(
    std::map<uint64_t, provider_info> hrd_providers)
    : hrd_providers_(std::move(hrd_providers)) {}

Offset<Provider> provider_builder::get_or_create_provider(
    uint64_t admin, FlatBufferBuilder& fbb) {
  return utl::get_or_create(fbs_providers_, admin, [&]() -> Offset<Provider> {
    auto it = hrd_providers_.find(admin);
    if (it == end(hrd_providers_)) {
      return 0;
    } else {
      return CreateProvider(
          fbb, to_fbs_string(fbb, it->second.short_name_, ENCODING),
          to_fbs_string(fbb, it->second.long_name_, ENCODING),
          to_fbs_string(fbb, it->second.full_name_, ENCODING));
    }
  });
}

}  // namespace motis::loader::hrd
