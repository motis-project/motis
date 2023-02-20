#pragma once

#include <string>

#include "flatbuffers/flatbuffers.h"

#include "rapidjson/document.h"

#include "motis/hash_map.h"
#include "motis/protocol/RISMessage_generated.h"
#include "motis/ris/ribasis/common.h"

namespace motis::ris::ribasis::fahrt {

struct context {
  explicit context(ris_msg_context& ris_ctx) : ris_{ris_ctx} {}

  ris_msg_context& ris_;

  mcd::hash_map<std::string, flatbuffers::Offset<StationInfo>> stations_;
  mcd::hash_map<std::string, flatbuffers::Offset<CategoryInfo>> categories_;
  mcd::hash_map<std::string, flatbuffers::Offset<flatbuffers::String>> lines_;
  mcd::hash_map<std::string, flatbuffers::Offset<ProviderInfo>> providers_;
};

void parse_ribasis_fahrt(ris_msg_context& ris_ctx,
                         rapidjson::Value const& data);

}  // namespace motis::ris::ribasis::fahrt
