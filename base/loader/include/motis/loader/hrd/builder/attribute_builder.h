#pragma once

#include <cinttypes>
#include <map>

#include "motis/loader/hrd/model/hrd_service.h"

#include "motis/loader/hrd/builder/bitfield_builder.h"

#include "motis/schedule-format/Attribute_generated.h"

namespace motis::loader::hrd {

struct attribute_builder {
  explicit attribute_builder(std::map<uint16_t, std::string> hrd_attributes);

  flatbuffers64::Offset<flatbuffers64::Vector<flatbuffers64::Offset<Attribute>>>
  create_attributes(std::vector<hrd_service::attribute> const&,
                    bitfield_builder&, flatbuffers64::FlatBufferBuilder&);

  flatbuffers64::Offset<Attribute> get_or_create_attribute(
      hrd_service::attribute const&, bitfield_builder&,
      flatbuffers64::FlatBufferBuilder&);

  std::map<uint16_t, std::string> hrd_attributes_;
  std::map<uint16_t, flatbuffers64::Offset<AttributeInfo>> fbs_attribute_infos_;
  std::map<std::pair<uint16_t, int>, flatbuffers64::Offset<Attribute>>
      fbs_attributes_;
};

}  // namespace motis::loader::hrd
