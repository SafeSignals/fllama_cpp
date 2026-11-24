# Prebuilt Vulkan shaders for Android (Veemoo fork)

fllama_cpp 1.1.0 ships no GPU backend on Android (ADR 004, NG2: "Reopens
when: a benchmark on a target Android device shows faster generation on the
GPU than on the CPU with a shipped model"). This fork builds ggml-vulkan for
Android arm64 so that benchmark can be run on phones.

ggml-vulkan normally compiles its shaders during the build, which needs glslc,
the Vulkan SDK and a compiler for the build machine besides the NDK. Here they
are compiled ahead of time instead:

- `shaders/`: the output of ggml's `vulkan-shaders-gen` (SPIR-V as C++
  arrays) for `src/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/*.comp`.
- `include/`: KhronosGroup Vulkan-Headers (`vulkan/`, `vk_video/`; Apache-2.0
  or MIT) and SPIR-V Headers (`spirv/unified1/spirv.hpp`; MIT-style Khronos
  licence).
- `features.cmake`: the glslc features the shaders were compiled with, and
  the glslc and header versions. ggml-vulkan.cpp is built with the same
  feature defines.

`src/CMakeLists.txt` points ggml-vulkan at this directory for Android, and
ggml-vulkan links the NDK's `libvulkan.so`. The library is bundled but loads
only when the app allows the GPU (`fllamaSetGpuAllowed(true)`).

**Regenerate after any change to ggml-vulkan** (a llama.cpp update), or the
shaders and the code that uses them won't match:

    tool/vulkan_prebuilt.sh GLSLC VULKAN_HEADERS_DIR SPIRV_HEADERS_DIR
