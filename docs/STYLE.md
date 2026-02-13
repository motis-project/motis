# MOTIS C++ Style

# Preamble

Beware that these rules only apply to MOTIS C++ and are very opinionated.
C++ has a big diversity of programming styles from "C with classes" to "Modern C++".
A lot of codebases have specific rules that make sense in this specific context
(e.g. embedded programming, gaming, Google search, etc.) and therefore different
guidelines. Over the years we learned that the rules described here are a good fit
for this specific project.

So in general our goals are:

- We want high-level, maintainable C++ code by default, not "high level assembly"
- but: don’t use features just because you can (like template meta programming, etc.)

# Style

- Header names: **`*.h`**, Implementation names: **`*.cc`**
- Don’t use include guards (`#ifndef #define #endif`), use **`#pragma once`**
- Consistently use **`struct`** instead of `class`
  - default visibility: public (which is what we need → no getter / setter)
  - you don’t need to write a constructor for 1-line initialization
- Always use ++i instead of i++ if it makes no difference for the program logic:
  `for (auto i = 0U; i < 10; ++i) { … }`
- Don't `using namespace std;`
- Don’t use `NULL` or `0`, use **nullptr** instead
- Don’t write `const auto&`, write **`auto const&`**
- Don’t write `const char*`, write **`char const*`**


# Case

- Everything **`snake_case`** (as in the C++ Standard Library)
- Template parameters **`PascalCase`** (as in the C++ Standard Library)
- Constants **`kPascalCase`** (as in the Google C++ Styleguide), not `UPPER_CASE` to prevent collisions with macro names
- Postfix **`member_variables_`** with an underscore to improve code readability when reading code without an IDE

```cpp
constexpr auto kMyConstant = 3.141;

template <typename TemplateType, int Size>
struct my_class : public my_parent {
  void member_fn(std::string const& fn_param) const override {
    auto const local_cvar = abc();
    auto local_var = def();
  }
  int my_field_;
};
```

# Includes

- Include only what you use (but everything you use!)
- Group includes:
  - for `.cc` files: first its own `.h` file
  - Standard headers with `<...>` syntax
    - C headers (use `<cstring>` instead of `<string.h>`, etc.)
    - C++ standard library headers (e.g. `<string>`)
  - Non-standard headers with `"..."` syntax
    - generic to specific = boost libraries, then more and more specific
    - last: project includes
    - if available: local includes `"./test_util.h"` from the local folder (only done for tests)
- Do not repeat include files from your own header file
- Repeat everything else - even it's transitiveley included already through other headers.
  The include might be removed from the header you include which leads broken compilation.
  Try to make the compilation as robust as possible. 


Example include files for `message.cc`:
```cpp
#include "motis/module/message.h"

#include <cstring>
#include <string>

#include "boost/asio.hpp"

#include "flatbuffers/idl.h"
#include "flatbuffers/util.h"

#include "motis/core/common/logging.h"
```

# Simplify Code: Predicate Usage

```cpp
// bad
if (is_valid()) {
  set_param(false);
} else {
  set_param(true);
}

// bad
set_param(is_valid() ? false : true);

// good
set_param(!is_valid());
```

# Always use Braces

```cpp
// bad
for (auto i = 0u; i < 10; ++i)
  if (is_valid())
    return get_a();
  else
    count_b();

// good
for (auto i = 0u; i < 10; ++i) {
  if (is_valid()) {
    return get_a();
  } else {
    count_b();
  }
}
```

# Use Short Variable Names

Only use shortened version of the variable name if it's still obvious what the variable holds.

- Index = `idx`
- Input = `in`
- Output = `out`
- Request = `req`
- Response = `res`
- Initialization = `init`
- ... etc.

If the context in which the variable is used is short, you can make variable names even shorter. For example `for (auto const& e : events) { /* ... */ }` or `auto const& b = get_buffer()`.

Don't use `lhs` and `rhs` - for comparison with `friend bool operator==`. Use `a` and `b`.

# Signatures in Headers

Omit information that's not needed for a forward declaration.

```cpp
void* get_memory(my_memory_manager& memory_manager); // bad

void* get_memory(my_memory_manager&); // good

// const for value parameters is not needed in headers
void calc_mask(bool const, bool const, bool const, bool const); // bad

void calc_mask(bool local_traffic, // slightly less bad
               bool long_distance_traffic,
               bool local_stations,
               bool long_distance_stations);

void calc_mask(mask_options); // good
```

# Low Indentation

Try to keep indentation at a minimum by handling cases one by one and bailing out early.

Example:

Bad:

```cpp
int main(int argc, char** argv) {
  if (argc > 1) {
    for (int i = 0; i < argc; ++i) {
      if (std::strcmp("hello", argv[i]) == 0) {
        /* ... 100 lines of code ... */
      }
    }
  }
}
```

Good:

```cpp
int main(int argc, char** argv) {
  if (argc <= 1) {
    return 0;
  }
  for (int i = 0; i < argc; ++i) {
    if (std::strcmp("hello", argv[i]) != 0) {
      continue;
    }
    /* ... 100 lines of code ... */
  }
}
```

# Function Length / File Length

Functions should have one task only. If they grow over ~50 lines of code, please check if they could be split into several functions to improve readability. But: don't split just randomly to not go over some arbitrary lines of code limit.

- Better: split earlier if it makes sense! Files are free! (more than one responsibility)
- Split later i.e. if you want to keep one block of logic without interruption (easier to understand)

# Pointers

Read C++ data types from right to left:

**`int const* const`** 
- `const` (read only) pointer (address can't be modified)
- to `const int` (int value at address can't be modified)

**int const&**
- reference
- on a const `int` value (read only)

**auto const&**
- reference
- on a value (type deduced by the compiler)

# Use RAII

Whenever possible use RAII to manage resource like memory (`std::unique_ptr`, `std::shared_ptr`, etc.), files (`std::fstream`), network sockets (Boost Asio), etc.

This means we do not want `new` or `delete` - except for placement new or placement delete in some very specific cases.

# Use `utl` Library

If there is no tool available in the C++ Standard Library please check first if we already have something in our [utl](https://github.com/motis-project/utl) library.

# Use `strong` types

Use `cista::strong` to define types, that cannot be converted implicitly. Using a `strong` type will ensure, that parameters cannot be mismatched, unlike `int` or `std::size_t`. This also makes function parameters clearer.

# `const`

Make everything (variables, loop variables, member functions, etc.) as `const` as possible. This indicates thread-safety (as long as only `const` methods are used) and helps to catch bugs when our mental model doesn't match the reality (the compiler will tell us).

# Initialization

Use [Aggregate Initialization](https://en.cppreference.com/w/cpp/language/aggregate_initialization) if possible. This also applies to member variables. A big advantage is that it doesn't allow implicit type conversions.

# Namespaces

Rename long namespace names instead of importing them completely.

```cpp
using boost::program_options;  // bad
namespace po = boost::program_options; // good
```

This way we still know where functions come from when reading code.
It becomes hard to know where a function came from when several large namespaces are completely imported.

Don't alias or import namespaces in header files.

# AAA-Style

Use [Almost Always Auto (AAA)](https://herbsutter.com/2013/08/12/gotw-94-solution-aaa-style-almost-always-auto/) style if possible.

- Program against interfaces
- Abstraction
- Less typing

Example: `for (auto const& el : c())`

No client code change if
- c returns another collection type (i.e. set instead of vector)
- the element type changes but still has a compatible interface

# No Raw Loops

It takes time to understand a raw for loop:
```cpp
for (int i = -1; i <= 9; i += 2) {
  if (i % 2 == 0) { continue; }
  if (i > 5 && i % 2 == 1) { break; }
  printf("%d\n", i/3);
}
```

- Raw for loops can
  - do crazy things
  - be boring (can often be expressed with a standard library algorithm!!)
  
- Find an element loop → `std::find`, `std::lower_bound`, ...
- Check each element loop → `std::all_of`, `std::none_of`, `std::any_of`
- Conversion loop → `std::transform`, `utl::to_vec`
- Counting: `std::count_if`, `std::accumulate`
- Sorting: `std::sort`, `std::nth_element`, `std::is_sorted`
- Logic: `std::all_of`, `std::any_of`
- Iterating multiple elements at once: `utl::zip`, `utl::pairwise`, `utl::nwise`
- Erasing elements: `utl::erase_if`, `utl::erase_duplicates`
- etc.

Hint: `utl` provides a cleaner interface wrapping `std::` functions for collections so you don't have to call `begin` and `end` all the time!

Benefits:
- Function name tells the reader of your code already what it does!
- Standard library implementation does not contain errors and is performant!

Alternative (if no function in the standard or `utl` helps):
- Use range based for loop if there's no named function: `for (auto const& el : collection) { .. }`

# Comparators

Either use
- Preferred: mark the operator you need `= default;`
- If that doesn't do the job you can check `CISTA_FRIEND_COMPARABLE`
- If you want to be selective and only compare a subset of member variables: `std::tie(a_, b_) == std::tie(a_, b_)`

# Set/Map vs Vector

Our go-to data structure is `std::vector`. (Hash-)maps and (hash-)sets are very expensive.

Never use `std::unordered_map`. We have better alternatives in all projects (e.g. unordered_dense).

## `vecvec` and `vector_map`

- Use `vector_map` for mappings with a `strong` key type and a continuous domain.
- Prefer using `vecvec<T>` instead of `vector<vector<T>>`, as data is stored and accessed more efficient. To store data, that may appear in any order, you may consider `paged_vecvec` instead.

# Tooling

- Always develop with Address Sanitizer (ASan) and Undefined Behaviour Sanitizer (UBSan) enabled if performance allows it (it's usually worth it to use small data sets to be able to develop with sanitizers enabled!): `CXXFLAGS=-fno-omit-frame-pointer -fsanitize=address,undefined`.
    - **Notice**: Some checks can cause false positive and should be disabled if necessary (compare `ci.yml`).  
      Example: `ASAN_OPTIONS=alloc_dealloc_mismatch=0`
- Check your code with `valgrind`.

# Spirit

- No deep inheritance hierarchies (no "enterprise" code)
- Don't write getters / setters for member variables: just make them public
  (which is the default for `struct` - remember: always use structs)
- Don't introduce a new variable for every value if it gets used only one time and the variable doesn't tell the reader any important information (-> inline variables).
- No GoF "design patterns" (Factory, Visitor, ...) if there is a simpler solution (there's always a simpler solution)
- Function / struct length:
  - it should be possible to understand every function by shortly looking at it
  - hints where to split:
    - single responsibility
    - short enough to be reusable in another context
- Don’t write “extensible” code that cares for functionality you might need at some point in the future. Just solve the problem at hand.
- Build the **smallest and simplest** solution possible that solves your problem
- Use abstractions to avoid thinking about details: helps to keep functions short
- Comment only the tricky / hacky pieces of your code
  (there should not be too many comments, otherwise your code is bad)
- Instead of comments use good (but short!) names for variables and functions
- Less code = less maintenance, less places for bugs, easier to understand
- Write robust code: `utl::verify()` assumptions about input data
