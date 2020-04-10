#pragma once

#include <cinttypes>
#include <map>

#include "motis/loader/hrd/parser/providers_parser.h"
#include "motis/schedule-format/Provider_generated.h"

namespace motis::loader::hrd {

struct provider_builder {
  explicit provider_builder(std::map<uint64_t, provider_info>);

  flatbuffers64::Offset<Provider> get_or_create_provider(
      uint64_t, flatbuffers64::FlatBufferBuilder&);

  std::map<uint64_t, provider_info> hrd_providers_;
  std::map<uint64_t, flatbuffers64::Offset<Provider>> fbs_providers_;
};

}  // namespace motis::loader::hrd
