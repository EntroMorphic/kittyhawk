# Glyph wrapper tests — verify the public glyph alias surface compiles,
# links, and dispatches to the live M4T substrate.

add_executable(test_glyph_wrapper test_glyph_wrapper.c)
target_link_libraries(test_glyph_wrapper PRIVATE glyph)
add_test(NAME glyph_wrapper COMMAND test_glyph_wrapper)

add_executable(test_glyph_libglyph test_glyph_libglyph.c)
target_link_libraries(test_glyph_libglyph PRIVATE glyph)
add_test(NAME glyph_libglyph COMMAND test_glyph_libglyph)

add_executable(test_routed_tool_smoke test_routed_tool_smoke.c)
add_test(NAME routed_tool_smoke COMMAND $<TARGET_FILE:test_routed_tool_smoke> $<TARGET_FILE:mnist_trit_lattice>)
