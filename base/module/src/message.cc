#include "motis/module/message.h"

#include <cstring>
#include <stdexcept>

#include "flatbuffers/idl.h"
#include "flatbuffers/util.h"

#include "utl/verify.h"

#include "motis/core/common/logging.h"
#include "motis/module/error.h"
#include "motis/module/fix_json.h"
#include "motis/protocol/resources.h"

#undef GetMessage

using namespace flatbuffers;

namespace motis::module {

std::unique_ptr<Parser> init_parser(
    json_format const jf = json_format::DEFAULT_FLATBUFFERS) {
  auto parser = std::make_unique<Parser>();
  parser->opts.strict_json = true;
  parser->opts.skip_unexpected_fields_in_json = true;
  parser->opts.output_default_scalars_in_json = true;

  switch (jf) {
    case json_format::DEFAULT_FLATBUFFERS: break;
    case json_format::SINGLE_LINE: parser->opts.indent_step = -1; break;
    case json_format::TYPES_IN_UNIONS:
    case json_format::CONTENT_ONLY_TYPES_IN_UNIONS:
      parser->opts.type_tags_in_unions = true;
      break;
  }

  int message_symbol_index = -1;
  for (unsigned i = 0; i < number_of_symbols; ++i) {
    if (strcmp(filenames[i], "Message.fbs") == 0) {  // NOLINT
      message_symbol_index = i;
    } else if (!parser->Parse(symbols[i], nullptr, filenames[i])) {  // NOLINT
      printf("error: %s\n", parser->error_.c_str());
      throw std::runtime_error("flatbuffer protocol definitions parser error");
    }
  }
  if (message_symbol_index == -1 ||
      !parser->Parse(symbols[message_symbol_index])) {  // NOLINT
    printf("error: %s\n", parser->error_.c_str());
    throw std::runtime_error("flatbuffer protocol definitions parser error");
  }
  return parser;
}

reflection::Schema const& init_schema(Parser& parser) {
  parser.Serialize();
  return *reflection::GetSchema(parser.builder_.GetBufferPointer());
}

namespace {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::unique_ptr<Parser> json_parser = init_parser();

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::unique_ptr<Parser> single_line_json_parser =
    init_parser(json_format::SINGLE_LINE);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::unique_ptr<Parser> types_in_unions_json_parser =
    init_parser(json_format::TYPES_IN_UNIONS);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::unique_ptr<Parser> reflection_parser = init_parser();

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
reflection::Schema const& schema = init_schema(*reflection_parser);

}  // namespace

std::string message::to_json(json_format const jf) const {
  std::string json;
  switch (jf) {
    case json_format::DEFAULT_FLATBUFFERS:
      flatbuffers::GenerateText(*json_parser, data(), &json);
      break;
    case json_format::SINGLE_LINE:
      flatbuffers::GenerateText(*single_line_json_parser, data(), &json);
      break;
    case json_format::TYPES_IN_UNIONS:
      flatbuffers::GenerateText(*types_in_unions_json_parser, data(), &json);
      break;
    case json_format::CONTENT_ONLY_TYPES_IN_UNIONS:
      flatbuffers::GenerateTextSingleField(*types_in_unions_json_parser, data(),
                                           &json, "content");
      break;
  }
  return json;
}

flatbuffers::StructDef const* get_fbs_struct_def(
    std::string const& table_name) {
  if (auto const it = json_parser->structs_.dict.find(table_name);
      it != json_parser->structs_.dict.end()) {
    return it->second;
  } else {
    throw utl::fail("fbs_table_to_json: table definition not found: {}",
                    table_name);
  }
}

std::string fbs_table_to_json(void const* table,
                              flatbuffers::StructDef const* struct_def,
                              json_format const jf) {
  std::string json;
  Parser* parser = nullptr;
  switch (jf) {
    case json_format::DEFAULT_FLATBUFFERS: parser = json_parser.get(); break;
    case json_format::SINGLE_LINE:
      parser = single_line_json_parser.get();
      break;
    case json_format::TYPES_IN_UNIONS:
      parser = types_in_unions_json_parser.get();
      break;
    case json_format::CONTENT_ONLY_TYPES_IN_UNIONS: break;
  }

  utl::verify(parser != nullptr, "fbs_table_to_json: invalid json format");

  flatbuffers::GenerateText(*parser, table, struct_def, &json);

  return json;
}

std::string fbs_table_to_json(void const* table, std::string const& table_name,
                              json_format const jf) {
  return fbs_table_to_json(table, get_fbs_struct_def(table_name), jf);
}

reflection::Schema const& message::get_schema() { return schema; }

reflection::Object const* message::get_objectref(char const* name) {
  return get_schema().objects()->LookupByKey(name);
}

std::pair<const char**, size_t> message::get_fbs_definitions() {
  return std::make_pair(symbols, number_of_symbols);
}

msg_ptr make_msg(std::string const& json, json_format& jf, bool const fix,
                 std::string_view const target, std::size_t const fbs_max_depth,
                 std::size_t const fbs_max_tables) {
  if (json.empty()) {
    LOG(motis::logging::error) << "empty request";
    throw std::system_error(error::unable_to_parse_msg);
  }

  auto parse_ok = false;
  if (fix) {
    auto const fix_result = fix_json(json, target);
    parse_ok = json_parser->Parse(fix_result.fixed_json_.c_str());
    jf = fix_result.detected_format_;
  } else {
    parse_ok = json_parser->Parse(json.c_str());
  }

  if (!parse_ok) {
    LOG(motis::logging::error)
        << "JSON parse error (step 2): " << json_parser->error_;
    throw std::system_error(error::unable_to_parse_msg);
  }

  flatbuffers::Verifier verifier(json_parser->builder_.GetBufferPointer(),
                                 json_parser->builder_.GetSize(), fbs_max_depth,
                                 fbs_max_tables);
  if (!VerifyMessageBuffer(verifier)) {
    LOG(motis::logging::error)
        << "JSON parse error (step 3): verification failed";
    throw std::system_error(error::malformed_msg);
  }
  auto size = json_parser->builder_.GetSize();
  auto buffer = json_parser->builder_.ReleaseBufferPointer();

  json_parser->builder_.Clear();
  return std::make_shared<message>(size, std::move(buffer));
}

msg_ptr make_msg(std::string const& json) {
  auto jf = json_format::DEFAULT_FLATBUFFERS;
  return make_msg(json, jf);
}

msg_ptr make_msg(message_creator& builder) {
  auto len = builder.GetSize();
  auto mem = builder.ReleaseBufferPointer();
  builder.Clear();
  return std::make_shared<message>(len, std::move(mem));
}

msg_ptr make_msg(void const* buf, size_t len, std::size_t const fbs_max_depth,
                 std::size_t const fbs_max_tables) {
  auto msg = std::make_shared<message>(len, buf);

  flatbuffers::Verifier verifier(msg->data(), msg->size(), fbs_max_depth,
                                 fbs_max_tables);
  if (!VerifyMessageBuffer(verifier)) {
    throw std::system_error(error::malformed_msg);
  }

  return msg;
}

msg_ptr make_no_msg(std::string const& target, int id) {
  message_creator b;
  b.create_and_finish(MsgContent_MotisNoMessage,
                      CreateMotisNoMessage(b).Union(), target,
                      DestinationType_Module, id);
  return make_msg(b);
}

msg_ptr make_success_msg(std::string const& target, int id) {
  message_creator b;
  b.create_and_finish(MsgContent_MotisSuccess, CreateMotisSuccess(b).Union(),
                      target, DestinationType_Module, id);
  return make_msg(b);
}

msg_ptr make_error_msg(std::error_code const& ec, int id) {
  message_creator b;
  b.create_and_finish(
      MsgContent_MotisError,
      CreateMotisError(b, ec.value(), b.CreateString(ec.category().name()),
                       b.CreateString(ec.message()))
          .Union(),
      "", DestinationType_Module, id);
  return make_msg(b);
}

msg_ptr make_unknown_error_msg(std::string const& reason, int id) {
  message_creator b;
  b.create_and_finish(
      MsgContent_MotisError,
      CreateMotisError(b, std::numeric_limits<uint16_t>::max(),
                       b.CreateString("unknown"), b.CreateString(reason))
          .Union(),
      "", DestinationType_Module, id);
  return make_msg(b);
}

}  // namespace motis::module
