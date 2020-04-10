#include "motis/loader/hrd/builder/category_builder.h"

#include "utl/get_or_create.h"

#include "utl/verify.h"

#include "motis/loader/hrd/parse_config.h"
#include "motis/loader/util.h"

namespace motis::loader::hrd {

using namespace utl;
using namespace flatbuffers64;

category_builder::category_builder(std::map<uint32_t, category> hrd_categories)
    : hrd_categories_(std::move(hrd_categories)) {}

Offset<Category> category_builder::get_or_create_category(
    cstr category_str, flatbuffers64::FlatBufferBuilder& fbb) {
  auto const category_key = raw_to_int<uint32_t>(category_str);
  auto it = hrd_categories_.find(category_key);
  utl::verify(it != end(hrd_categories_), "missing category: {}",
              category_str.view());

  return utl::get_or_create(fbs_categories_, category_key, [&]() {
    return CreateCategory(fbb, to_fbs_string(fbb, it->second.name_, ENCODING),
                          it->second.output_rule_);
  });
}

}  // namespace motis::loader::hrd
