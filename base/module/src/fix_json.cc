#include "motis/module/fix_json.h"

#include <cstring>
#include <set>
#include <string_view>

#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

#include "motis/module/error.h"

using namespace rapidjson;

namespace motis::module {

namespace {

constexpr auto const kTypeKey = "_type";

}  // namespace

void write_json_value(Value const& v, Writer<StringBuffer>& writer,
                      std::string_view current_key = std::string_view{},
                      bool const is_root = false,
                      std::string_view const target = std::string_view{}) {
  auto const is_type_key = [](auto const& m, std::string_view& union_key) {
    auto const key = m.name.GetString();
    auto const key_len = m.name.GetStringLength();
    if (key_len > 5 && std::strcmp(key + key_len - 5, "_type") == 0) {
      union_key = {key, key_len - 5};
      return true;
    }
    return false;
  };

  auto add_msg_wrapper = false;
  if (is_root && v.IsObject()) {
    if (!v.HasMember("destination") && !v.HasMember("content")) {
      add_msg_wrapper = true;
      writer.StartObject();

      writer.String("destination");
      writer.StartObject();
      writer.String("target");
      writer.String(target.data(), static_cast<SizeType>(target.size()));
      writer.EndObject();

      current_key = "content";
    }
  }

  if (!current_key.empty()) {
    if (v.IsObject()) {
      if (auto const it = v.GetObject().FindMember(kTypeKey);
          it != v.MemberEnd()) {
        auto const union_key = std::string{current_key} + kTypeKey;
        writer.String(union_key.data(),
                      static_cast<SizeType>(union_key.size()));
        write_json_value(it->value, writer);
      }
    }
    writer.String(current_key.data(),
                  static_cast<SizeType>(current_key.size()));
  }

  switch (v.GetType()) {  // NOLINT
    case rapidjson::kObjectType: {
      writer.StartObject();

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
        writer.String(m.name.GetString(), m.name.GetStringLength());
        write_json_value(m.value, writer);

        // Write union.
        write_json_value(it->value, writer, union_key);

        // Remember written values.
        written.emplace(m.name.GetString(), m.name.GetStringLength());
        written.emplace(union_key);
      }

      // Write remaining values
      for (auto const& m : v.GetObject()) {
        auto const key =
            std::string_view{m.name.GetString(), m.name.GetStringLength()};
        if (key != kTypeKey && written.find(key) == end(written)) {
          write_json_value(m.value, writer, key);
        }
      }

      // Write message id if missing.
      if (is_root && !v.HasMember("id")) {
        writer.String("id");
        writer.Int(1);
      }

      writer.EndObject();
      break;
    }

    case rapidjson::kArrayType: {
      writer.StartArray();
      for (auto const& entry : v.GetArray()) {
        write_json_value(entry, writer);
      }
      writer.EndArray();
      break;
    }

    default: v.Accept(writer);  // NOLINT
  }

  if (add_msg_wrapper) {
    writer.EndObject();
  }
}

std::string fix_json(std::string const& json, std::string_view const target) {
  rapidjson::Document d;
  if (d.Parse(json.c_str()).HasParseError()) {  // NOLINT
    throw std::system_error(module::error::unable_to_parse_msg);
  }

  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);

  write_json_value(d, writer, std::string_view{}, true, target);

  return buffer.GetString();
}

}  // namespace motis::module
