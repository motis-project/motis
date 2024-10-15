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

void write_json_value(Value const& v, Writer<StringBuffer>& writer,
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
        writer.String(it->name.GetString(), it->name.GetStringLength());
        write_json_value(it->value, writer);

        // Remember written values.
        written.emplace(m.name.GetString(), m.name.GetStringLength());
        written.emplace(union_key);
      }

      // Write remaining values
      for (auto const& m : v.GetObject()) {
        if (written.find({m.name.GetString(), m.name.GetStringLength()}) ==
            end(written)) {
          writer.String(m.name.GetString(), m.name.GetStringLength());
          write_json_value(m.value, writer);
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
}

std::string fix_json(std::string const& json) {
  rapidjson::Document d;
  if (d.Parse(json.c_str()).HasParseError()) {  // NOLINT
    throw std::system_error(module::error::unable_to_parse_msg);
  }

  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);

  write_json_value(d, writer, true);

  return buffer.GetString();
}

}  // namespace motis::module
