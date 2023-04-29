#include "motis/module/fix_json.h"

#include <cstring>
#include <set>
#include <string_view>

#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

#include "motis/core/common/logging.h"
#include "motis/module/error.h"

using namespace rapidjson;

namespace motis::module {

struct json_converter {
  static constexpr auto const kTypeKey = "_type";

  json_converter(Writer<StringBuffer>& writer, std::string_view target)
      : writer_{writer}, target_{target} {}

  void fix(Value const& v) { write_json_value(v, std::string_view{}, true); }

  json_format detected_format() const {
    if (content_only_detected_) {
      return json_format::CONTENT_ONLY_TYPES_IN_UNIONS;
    } else if (type_in_union_detected_) {
      return json_format::TYPES_IN_UNIONS;
    } else {
      return json_format::DEFAULT_FLATBUFFERS;
    }
  }

private:
  void write_json_value(Value const& v,
                        std::string_view current_key = std::string_view{},
                        bool const is_root = false) {
    auto const is_type_key = [](auto const& m, std::string_view& union_key) {
      auto const key = m.name.GetString();
      auto const key_len = m.name.GetStringLength();
      if (key_len > 5 && std::strcmp(key + key_len - 5, "_type") == 0) {
        union_key = {key, key_len - 5};
        return true;
      }
      return false;
    };

    // Check if the root element has destination + content fields
    // or uses the compact format (content only, target taken from URL).
    auto add_msg_wrapper = false;
    if (is_root && v.IsObject()) {
      if (!v.HasMember("destination") && !v.HasMember("content")) {
        add_msg_wrapper = true;
        content_only_detected_ = true;
        writer_.StartObject();

        writer_.String("destination");
        writer_.StartObject();
        writer_.String("target");
        writer_.String(target_.data(), static_cast<SizeType>(target_.size()));
        writer_.EndObject();

        current_key = "content";
      }
    }

    if (!current_key.empty()) {
      if (v.IsObject()) {
        // We're inside an object field. Check if it has a type key.
        if (auto const it = v.GetObject().FindMember(kTypeKey);
            it != v.MemberEnd()) {
          // Type key found, write it before the object field itself.
          auto const union_key = std::string{current_key} + kTypeKey;
          writer_.String(union_key.data(),
                         static_cast<SizeType>(union_key.size()));
          write_json_value(it->value);
          type_in_union_detected_ = true;
        }
      }
      writer_.String(current_key.data(),
                     static_cast<SizeType>(current_key.size()));
    }

    switch (v.GetType()) {  // NOLINT
      case rapidjson::kObjectType: {
        writer_.StartObject();

        // Set of already written members.
        std::set<std::string_view> written;

        for (auto const& m : v.GetObject()) {
          std::string_view union_key;
          if (!is_type_key(m, union_key)) {
            continue;  // Not a union key.
          }

          auto const it = v.GetObject().FindMember(
              Value(union_key.data(), static_cast<SizeType>(union_key.size())));
          if (it == v.MemberEnd()) {
            continue;  // Could be a union key but no union found.
          }

          // Write union key ("_type").
          writer_.String(m.name.GetString(), m.name.GetStringLength());
          write_json_value(m.value);

          // Write union.
          write_json_value(it->value, union_key);

          // Remember written values.
          written.emplace(m.name.GetString(), m.name.GetStringLength());
          written.emplace(union_key);
        }

        // Write remaining values.
        for (auto const& m : v.GetObject()) {
          auto const key =
              std::string_view{m.name.GetString(), m.name.GetStringLength()};
          if (key != kTypeKey && written.find(key) == end(written)) {
            write_json_value(m.value, key);
          }
        }

        // Write message id if missing.
        if (is_root && !v.HasMember("id")) {
          writer_.String("id");
          writer_.Int(1);
        }

        writer_.EndObject();
        break;
      }

      case rapidjson::kArrayType: {
        writer_.StartArray();
        for (auto const& entry : v.GetArray()) {
          write_json_value(entry);
        }
        writer_.EndArray();
        break;
      }

      default: v.Accept(writer_);  // NOLINT
    }

    if (add_msg_wrapper) {
      writer_.EndObject();
    }
  }

  Writer<StringBuffer>& writer_;  // NOLINT
  std::string_view target_;
  bool content_only_detected_{false};
  bool type_in_union_detected_{false};
};

fix_json_result fix_json(std::string const& json,
                         std::string_view const target) {
  rapidjson::Document d;
  if (d.Parse(json.c_str()).HasParseError()) {  // NOLINT
    LOG(motis::logging::error)
        << "JSON parse error (step 1): " << GetParseError_En(d.GetParseError())
        << " at offset " << d.GetErrorOffset();
    throw std::system_error(module::error::unable_to_parse_msg);
  }

  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);

  auto converter = json_converter{writer, target};
  converter.fix(d);
  return {buffer.GetString(), converter.detected_format()};
}

}  // namespace motis::module
