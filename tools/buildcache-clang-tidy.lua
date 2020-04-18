-- match(.*cmake.*)

-------------------------------------------------------------------------------
-- This is a re-implementation of the C++ class gcc_wrapper_t.
-------------------------------------------------------------------------------

require_std("io")
require_std("os")
require_std("string")
require_std("table")
require_std("bcache")


-------------------------------------------------------------------------------
-- Internal helper functions.
-------------------------------------------------------------------------------

local function make_preprocessor_cmd (args, preprocessed_file)
  local preprocess_args = {}

  -- Drop arguments that we do not want/need.
  local drop_next_arg = false
  for i, arg in ipairs(args) do
    local drop_this_arg = drop_next_arg
    drop_next_arg = false
    if arg == "-c" then
      drop_this_arg = true
    elseif arg == "-o" then
      drop_this_arg = true
      drop_next_arg = true
    end
    if not drop_this_arg and i > 6 then
      table.insert(preprocess_args, arg)
    end
  end

  -- Append the required arguments for producing preprocessed output.
  table.insert(preprocess_args, "-E")
  table.insert(preprocess_args, "-P")
  table.insert(preprocess_args, "-o")
  table.insert(preprocess_args, preprocessed_file)

  return preprocess_args
end

local function is_source_file (path)
  local ext = bcache.get_extension(path):lower()
  return (ext == ".cpp") or (ext == ".cc") or (ext == ".cxx") or (ext == ".c")
end


-------------------------------------------------------------------------------
-- Wrapper interface implementation.
-------------------------------------------------------------------------------

function get_capabilities ()
  -- We can use hard links with GCC since it will never overwrite already
  -- existing files.
  return { "hard_links" }
end

function preprocess_source ()
  local expected = {"-E", "__run_co_compile", "--tidy", "--source", "--"}
  for i, arg in ipairs(expected) do
    if string.sub(ARGS[i+1], 1, #expected[i]) ~= expected[i] then
      error("Unexpected argument " .. i .. ": " .. ARGS[i+1] .. " -> '" .. string.sub(ARGS[i+1], 1, 1 + #expected[i]) .. "' != '" .. expected[i] .. "'")
    end
  end

  -- Check if this is a compilation command that we support.
  local is_object_compilation = false
  local has_object_output = false
  for i, arg in ipairs(ARGS) do
    if arg == "-c" then
      is_object_compilation = true
    elseif arg == "-o" then
      has_object_output = true
    elseif arg:sub(1, 1) == "@" then
      error("Response files are currently not supported.")
    end
  end
  if (not is_object_compilation) or (not has_object_output) then
    error("Unsupported complation command.")
  end

  -- Run the preprocessor step.
  local preprocessed_file = os.tmpname()
  local preprocessor_args = make_preprocessor_cmd(ARGS, preprocessed_file)

  local result = bcache.run(preprocessor_args)
  if result.return_code ~= 0 then
    os.remove(preprocessed_file)
    error("Preprocessing command was unsuccessful.")
  end

  -- Read and return the preprocessed file.
  local f = assert(io.open(preprocessed_file, "rb"))
  local preprocessed_source = f:read("*all")
  f:close()
  os.remove(preprocessed_file)

  return preprocessed_source
end

function get_relevant_arguments ()
  local filtered_args = {}

  -- The first argument is the compiler binary without the path.
  table.insert(filtered_args, bcache.get_file_part(ARGS[1]))

  -- Note: We always skip the first arg since we have handled it already.
  local skip_next_arg = true
  for i, arg in ipairs(ARGS) do
    if not skip_next_arg then
      -- Does this argument specify a file (we don't want to hash those).
      local is_arg_plus_file_name = (arg == "-I") or (arg == "-MF") or
                                    (arg == "-MT") or (arg == "-MQ") or
                                    (arg == "-o")

      -- Generally unwanted argument (things that will not change how we go
      -- from preprocessed code to binary object files)?
      local first_two_chars = arg:sub(1, 2)
      local is_unwanted_arg = (first_two_chars == "-I") or
                              (first_two_chars == "-D") or
                              (first_two_chars == "-M") or
                              is_source_file(arg)

      if is_arg_plus_file_name then
        skip_next_arg = true
      elseif not is_unwanted_arg then
        table.insert(filtered_args, arg)
      end
    else
      skip_next_arg = false
    end
  end

  return filtered_args
end

function get_program_id ()
  -- TODO(m): Add things like executable file size too.

  -- Get the version string for the compiler.
  local result = bcache.run({ARGS[1], "--version"})
  if result.return_code ~= 0 then
    error("Unable to get the compiler version information string.")
  end

  return result.std_out
end

function get_build_files ()
  local files = {}
  local found_object_file = false
  for i = 2, #ARGS do
    local next_idx = i + 1
    if (ARGS[i] == "-o") and (next_idx <= #ARGS) then
      if found_object_file then
        error("Only a single target object file can be specified.")
      end
      files["object"] = ARGS[next_idx]
      found_object_file = true
    elseif (ARGS[i] == "-ftest-coverage") then
      error("Code coverage data is currently not supported.")
    end
  end
  if not found_object_file then
    error("Unable to get the target object file.")
  end
  return files
end