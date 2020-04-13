/*
 Copyright (c) 2011, Micael Hildenborg
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Micael Hildenborg nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY Micael Hildenborg ''AS IS'' AND ANY
 EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL Micael Hildenborg BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 Contributors:
 Gustav
 Several members in the gamedev.se forum.
 Gregory Petrosyan
 */

#pragma once

#include <cinttypes>
#include <array>
#include <string>
#include <string_view>
#include <vector>

namespace sha1 {

using hash_t = std::array<std::uint8_t, 20>;

// Rotate an integer value to left.
inline unsigned rol(unsigned const value, unsigned const steps) {
  return ((value << steps) | (value >> (32 - steps)));
}

// Sets the first 16 integers in the buffert to zero.
// Used for clearing the W buffert.
inline void clear_w_buffert(unsigned* buffert) {
  for (int pos = 16; --pos >= 0;) {
    buffert[pos] = 0;
  }
}

inline void inner_hash(unsigned* result, unsigned* w) {
  struct data {
    explicit data(unsigned* r)
        : a_{r[0]}, b_{r[1]}, c_{r[2]}, d_{r[3]}, e_{r[4]} {}

    inline void macro(unsigned const func, unsigned const val,
                      unsigned const round, unsigned const* w) {
      const unsigned t = rol(a_, 5) + (func) + e_ + val + w[round];
      e_ = d_;
      d_ = c_;
      c_ = rol(b_, 30);
      b_ = a_;
      a_ = t;
    }

    unsigned a_;
    unsigned b_;
    unsigned c_;
    unsigned d_;
    unsigned e_;
  } x(result);

  int round = 0;

  while (round < 16) {
    x.macro((x.b_ & x.c_) | (~x.b_ & x.d_), 0x5a827999, round, w);
    ++round;
  }
  while (round < 20) {
    w[round] =
        rol((w[round - 3] ^ w[round - 8] ^ w[round - 14] ^ w[round - 16]), 1);
    x.macro((x.b_ & x.c_) | (~x.b_ & x.d_), 0x5a827999, round, w);
    ++round;
  }
  while (round < 40) {
    w[round] =
        rol((w[round - 3] ^ w[round - 8] ^ w[round - 14] ^ w[round - 16]), 1);
    x.macro(x.b_ ^ x.c_ ^ x.d_, 0x6ed9eba1, round, w);
    ++round;
  }
  while (round < 60) {
    w[round] =
        rol((w[round - 3] ^ w[round - 8] ^ w[round - 14] ^ w[round - 16]), 1);
    x.macro((x.b_ & x.c_) | (x.b_ & x.d_) | (x.c_ & x.d_), 0x8f1bbcdc, round,
            w);
    ++round;
  }
  while (round < 80) {
    w[round] =
        rol((w[round - 3] ^ w[round - 8] ^ w[round - 14] ^ w[round - 16]), 1);
    x.macro(x.b_ ^ x.c_ ^ x.d_, 0xca62c1d6, round, w);
    ++round;
  }

  result[0] += x.a_;
  result[1] += x.b_;
  result[2] += x.c_;
  result[3] += x.d_;
  result[4] += x.e_;
}

inline hash_t sha1(std::string_view const& src) {
  hash_t hash;

  // Init the result array.
  unsigned result[5] = {0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476,
                        0xc3d2e1f0};

  // Cast the void src pointer to be the byte array we can work with.
  auto const sarray = reinterpret_cast<uint8_t const*>(src.data());

  // The reusable round buffer
  unsigned w[80];

  // Loop through all complete 64byte blocks.
  size_t const end_of_full_blocks = src.size() - 64;
  size_t end_of_current_block;
  size_t current_block = 0;

  while (current_block <= end_of_full_blocks) {
    end_of_current_block = current_block + 64;

    // Init the round buffer with the 64 byte block data.
    for (int round_pos = 0; current_block < end_of_current_block;
         current_block += 4) {
      // This line will swap endian on big endian and keep endian on little
      // endian.
      w[round_pos++] =
          static_cast<unsigned>(sarray[current_block + 3U]) |
          (static_cast<unsigned>(sarray[current_block + 2U]) << 8U) |
          (static_cast<unsigned>(sarray[current_block + 1U]) << 16U) |
          (static_cast<unsigned>(sarray[current_block]) << 24U);
    }
    inner_hash(result, w);
  }

  // Handle the last and not full 64 byte block if existing.
  end_of_current_block = src.size() - current_block;
  clear_w_buffert(w);
  auto last_block_bytes = 0UL;
  for (; last_block_bytes < end_of_current_block; ++last_block_bytes) {
    w[last_block_bytes >> 2U] |=
        static_cast<unsigned>(sarray[last_block_bytes + current_block])
        << ((3U - (last_block_bytes & 3U)) << 3U);
  }
  w[last_block_bytes >> 2U] |= 0x80U << ((3U - (last_block_bytes & 3U)) << 3U);
  if (end_of_current_block >= 56U) {
    inner_hash(result, w);
    clear_w_buffert(w);
  }

  // XXX possible overflow for src.size() > 4GB?
  w[15] = static_cast<unsigned>(src.size()) << 3U;
  inner_hash(result, w);

  // Store hash in result pointer, and make sure we get in in the correct order
  // on both endian models.
  for (auto hash_byte = 20; --hash_byte >= 0;) {
    auto hb = static_cast<unsigned>(hash_byte);
    hash[hb] = (result[hb >> 2U] >> (((3U - hb) & 0x3U) << 3U)) & 0xffU;
    if (hb == 0U) {
      break;
    }
  }

  return hash;
}

inline std::string to_hex_str(hash_t const& hash) {
  const char hexDigits[] = {"0123456789abcdef"};

  std::string hexstring;
  hexstring.resize(40);
  for (auto hash_byte = 20; --hash_byte >= 0;) {
    auto const hb = static_cast<unsigned>(hash_byte);
    hexstring[hb << 1U] = hexDigits[(hash[hb] >> 4U) & 0xfU];
    hexstring[(hb << 1U) + 1U] = hexDigits[hash[hb] & 0xfU];
    if (hb == 0U) {
      break;
    }
  }
  hexstring[40] = 0U;
  return hexstring;
}

inline std::string from_buf(std::vector<std::uint8_t> const& buf) {
  return to_hex_str(sha1({reinterpret_cast<char const*>(&buf[0]), buf.size()}));
}

}  // namespace sha1