#pragma once

#include <cstdint>

namespace trix_native {

// Encode ternary {-1,0,+1} into 2-bit codes:
//   00 -> 0
//   01 -> +1
//   10 -> -1
// This matches the encoding used elsewhere in the repo.
static inline uint8_t ternary_code(int8_t v) {
  if (v > 0) return 0x01;
  if (v < 0) return 0x02;
  return 0x00;
}

// Pack 4 ternary codes into one byte (2 bits each).
static inline uint8_t pack4(int8_t a0, int8_t a1, int8_t a2, int8_t a3) {
  return static_cast<uint8_t>(
      (ternary_code(a0) << 0) | (ternary_code(a1) << 2) | (ternary_code(a2) << 4) |
      (ternary_code(a3) << 6));
}

static inline int popcount8(uint8_t x) {
  return __builtin_popcount(static_cast<unsigned int>(x));
}

// Build a bitmask that keeps only coordinates where x is non-zero.
// For each 2-bit group:
//   code 00 -> mask 00
//   code 01/10 -> mask 11
static inline uint8_t mask_nonzero_groups(uint8_t packed_x) {
  uint8_t m = 0;
  if (packed_x & 0x03) m |= 0x03;
  if (packed_x & 0x0C) m |= 0x0C;
  if (packed_x & 0x30) m |= 0x30;
  if (packed_x & 0xC0) m |= 0xC0;
  return m;
}

// Popcount distance between packed ternary vectors.
// If ignore_x_zeros is true, coordinates where x==0 (code 00) contribute 0.
static inline int popcount_distance_packed(
    const uint8_t* packed_x,
    const uint8_t* packed_s,
    int packed_bytes,
    bool ignore_x_zeros) {
  int d = 0;
  if (!ignore_x_zeros) {
    for (int i = 0; i < packed_bytes; i++) {
      d += popcount8(static_cast<uint8_t>(packed_x[i] ^ packed_s[i]));
    }
    return d;
  }

  for (int i = 0; i < packed_bytes; i++) {
    const uint8_t mx = mask_nonzero_groups(packed_x[i]);
    d += popcount8(static_cast<uint8_t>((packed_x[i] ^ packed_s[i]) & mx));
  }
  return d;
}

}  // namespace trix_native
