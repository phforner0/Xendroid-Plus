/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2013 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/cpu/ppc/ppc_scanner.h"

#include <algorithm>
#include <vector>

#include "xenia/base/byte_order.h"
#include "xenia/base/cvar.h"
#include "xenia/base/logging.h"
#include "xenia/base/memory.h"
#include "xenia/base/profiling.h"
#include "xenia/cpu/ppc/ppc_decode_data.h"
#include "xenia/cpu/ppc/ppc_frontend.h"
#include "xenia/cpu/ppc/ppc_jump_table.h"
#include "xenia/cpu/ppc/ppc_opcode_info.h"
#include "xenia/cpu/processor.h"
#include "xenia/cpu/xex_module.h"

DEFINE_bool(ppc_scanner_jump_tables, true,
            "Recover static MSVC jump tables (mtctr/bctr) in PPCScanner.",
            "CPU");

#if 0
#define LOGPPC(fmt, ...) XELOGCORE('p', fmt, ##__VA_ARGS__)
#else
#define LOGPPC(fmt, ...) \
  do {                   \
  } while (false)
#endif

namespace xe {
namespace cpu {
namespace ppc {

namespace {

struct JumpTableReaderContext {
  Memory* memory;
  Module* module;
  XexModule* xex_module;
};

bool CanReadJumpTableAddress(const JumpTableReaderContext* ctx, uint32_t addr,
                             uint32_t size, const uint8_t** out_ptr) {
  if (!ctx || !ctx->memory || !ctx->module || !out_ptr) {
    return false;
  }
  if (size == 0 || addr > UINT32_MAX - (size - 1u)) {
    return false;
  }
  const uint32_t end_addr = addr + (size - 1u);
  if ((addr >= 0x7FC00000u && addr < 0x80000000u) ||
      (end_addr >= 0x7FC00000u && end_addr < 0x80000000u)) {
    return false;
  }
  if (ctx->memory->LookupVirtualMappedRange(addr) ||
      ctx->memory->LookupVirtualMappedRange(end_addr)) {
    return false;
  }
  BaseHeap* start_heap = ctx->memory->LookupHeap(addr);
  BaseHeap* end_heap = ctx->memory->LookupHeap(end_addr);
  if (!start_heap || start_heap != end_heap) {
    return false;
  }
  const auto access = start_heap->QueryRangeAccess(addr, end_addr);
  if (access == xe::memory::PageAccess::kNoAccess) {
    return false;
  }
  if (ctx->xex_module) {
    const bool in_module = ctx->xex_module->ContainsAddress(addr) &&
                           ctx->xex_module->ContainsAddress(end_addr);
    if (!in_module && access != xe::memory::PageAccess::kReadOnly) {
      return false;
    }
  }
  if (start_heap->heap_type() == HeapType::kGuestPhysical) {
    auto* phys_heap = static_cast<PhysicalHeap*>(start_heap);
    const uint32_t phys_start = phys_heap->GetPhysicalAddress(addr);
    const uint32_t phys_end = phys_heap->GetPhysicalAddress(end_addr);
    if (phys_start == UINT32_MAX || phys_end == UINT32_MAX ||
        phys_end - phys_start != size - 1u) {
      return false;
    }
    *out_ptr = ctx->memory->TranslatePhysical(phys_start);
    return *out_ptr != nullptr;
  }
  *out_ptr = ctx->memory->TranslateVirtual<const uint8_t*>(addr);
  return *out_ptr != nullptr;
}

bool JumpTableReadU8(void* raw_ctx, uint32_t addr, uint8_t* out) {
  const auto* ctx = static_cast<const JumpTableReaderContext*>(raw_ctx);
  const uint8_t* ptr = nullptr;
  if (!out || !CanReadJumpTableAddress(ctx, addr, 1u, &ptr)) {
    return false;
  }
  *out = *ptr;
  return true;
}

bool JumpTableReadU16BE(void* raw_ctx, uint32_t addr, uint16_t* out) {
  const auto* ctx = static_cast<const JumpTableReaderContext*>(raw_ctx);
  const uint8_t* ptr = nullptr;
  if (!out || !CanReadJumpTableAddress(ctx, addr, 2u, &ptr)) {
    return false;
  }
  *out = xe::load_and_swap<uint16_t>(ptr);
  return true;
}

bool JumpTableReadU32BE(void* raw_ctx, uint32_t addr, uint32_t* out) {
  const auto* ctx = static_cast<const JumpTableReaderContext*>(raw_ctx);
  const uint8_t* ptr = nullptr;
  if (!out || !CanReadJumpTableAddress(ctx, addr, 4u, &ptr)) {
    return false;
  }
  *out = xe::load_and_swap<uint32_t>(ptr);
  return true;
}

bool JumpTableIsValidCodeTarget(void* raw_ctx, uint32_t target_addr) {
  const auto* ctx = static_cast<const JumpTableReaderContext*>(raw_ctx);
  if (!ctx || !ctx->module) {
    return false;
  }
  if (ctx->xex_module) {
    if (!ctx->xex_module->ContainsAddress(target_addr)) {
      return false;
    }
    xex2_section_type section_type{};
    if (!ctx->xex_module->GetPageSectionType(target_addr, &section_type)) {
      return false;
    }
    return section_type == XEX_SECTION_CODE;
  }
  return ctx->module->ContainsAddress(target_addr);
}

}  // namespace

PPCScanner::PPCScanner(PPCFrontend* frontend) : frontend_(frontend) {}

PPCScanner::~PPCScanner() {}

bool PPCScanner::IsRestGprLr(uint32_t address) {
  auto function = frontend_->processor()->QueryFunction(address);
  return function && function->behavior() == Function::Behavior::kEpilogReturn;
}

void PPCScanner::TryRecoverJumpTableAt(GuestFunction* function,
                                       uint32_t start_address,
                                       uint32_t bctr_address,
                                       uint32_t* furthest_target) {
  if (!cvars::ppc_scanner_jump_tables) {
    return;
  }
  Module* module = function->module();
  if (!module || frontend_->processor()->IsDynamicCodeModule(module)) {
    return;
  }

  const uint32_t max_window_bytes = (kMaxJumpTableBacktrackInsns - 1u) * 4u;
  const uint32_t window_start =
      (bctr_address > start_address &&
       (bctr_address - start_address) > max_window_bytes)
          ? (bctr_address - max_window_bytes)
          : start_address;

  std::vector<uint32_t> window;
  window.reserve(kMaxJumpTableBacktrackInsns);
  for (uint32_t a = window_start; a <= bctr_address; a += 4u) {
    const uint8_t* ptr = module->TranslateCode(a);
    if (!ptr) {
      return;
    }
    window.push_back(xe::load_and_swap<uint32_t>(ptr));
  }
  if (window.empty()) {
    return;
  }

  JumpTableReaderContext reader_ctx{
      module->memory(),
      module,
      dynamic_cast<XexModule*>(module),
  };
  JumpTableMemoryReader reader{
      &reader_ctx,
      &JumpTableReadU8,
      &JumpTableReadU16BE,
      &JumpTableReadU32BE,
      &JumpTableIsValidCodeTarget,
  };

  JumpTableInfo table;
  if (!TryRecoverJumpTable(start_address, bctr_address, window.data(),
                           window.size(), reader, &table)) {
    return;
  }

  std::vector<uint32_t> unique_targets = std::move(table.targets);
  std::sort(unique_targets.begin(), unique_targets.end());
  unique_targets.erase(
      std::unique(unique_targets.begin(), unique_targets.end()),
      unique_targets.end());
  for (uint32_t target : unique_targets) {
    *furthest_target = std::max(*furthest_target, target);
  }
  function->AddJumpTable(bctr_address, std::move(unique_targets));
}

bool PPCScanner::Scan(GuestFunction* function, FunctionDebugInfo* debug_info) {
  // This is a simple basic block analyizer. It walks the start address to the
  // end address looking for branches. Each span of instructions between
  // branches is considered a basic block. When the last blr (that has no
  // branches to after it) is found the function is considered ended. If this
  // is before the expected end address then the function address range is
  // split up and the second half is treated as another function.

  Module* module = function->module();
  function->ClearJumpTables();

  LOGPPC("Analyzing function {:08X}...", function->address());

  // For debug info, only if needed.
  uint32_t address_reference_count = 0;
  uint32_t instruction_result_count = 0;

  uint32_t start_address = static_cast<uint32_t>(function->address());
  uint32_t end_address = static_cast<uint32_t>(function->end_address());
  uint32_t address = start_address;
  // Every exit leaves address somewhere different, so track what the walk took.
  uint32_t last_address = 0;
  uint32_t furthest_target = start_address;
  size_t blocks_found = 0;
  bool in_block = false;
  bool starts_with_mfspr_lr = false;
  while (true) {
    // Checked before the read, not after the step, so the bound holds even
    // when it lands below the address the walk started from.
    if (end_address && address > end_address) {
      LOGPPC("Ran over function bounds! {:08X}-{:08X}", start_address,
             end_address);
      break;
    }
    // Dynamic code is backed by whatever the guest committed, so stop rather
    // than read past it. The first page holds the function start.
    if (!(address & 0xFFF) && !module->ContainsAddress(address)) {
      LOGPPC("function end {:08X} (outside the module)", address);
      break;
    }
    uint32_t code = xe::load_and_swap<uint32_t>(module->TranslateCode(address));

    // If we fetched 0 assume that we somehow hit one of the awesome
    // 'no really we meant to end after that bl' functions.
    if (!code) {
      LOGPPC("function end {:08X} (0x00000000 read)", address);
      break;
    }

    last_address = address;

    auto opcode = LookupOpcode(code);

    PPCDecodeData d;
    d.address = address;
    d.code = code;

    // TODO(benvanik): switch on instruction metadata.
    ++address_reference_count;
    ++instruction_result_count;

    // Check if the function starts with a mfspr lr, as that's a good indication
    // of whether or not this is a normal function with a prolog/epilog.
    // Some valid leaf functions won't have this, but most will.
    if (address == start_address && opcode == PPCOpcode::mfspr &&
        (((d.XFX.SPR() & 0x1F) << 5) | ((d.XFX.SPR() >> 5) & 0x1F)) == 8) {
      starts_with_mfspr_lr = true;
    }

    if (!in_block) {
      in_block = true;
      blocks_found++;
    }

    bool ends_fn = false;
    bool ends_block = false;
    if (opcode == PPCOpcode::kInvalid) {
      // Invalid instruction.
      // We can just ignore it because there's (very little)/no chance it'll
      // affect flow control.
      LOGPPC("Invalid instruction at {:08X}: {:08X}", address, code);
    } else if (code == 0x4E800020) {
      // blr -- unconditional branch to LR.
      // This is generally a return.
      if (furthest_target > address) {
        // Remaining targets within function, not end.
        LOGPPC("ignoring blr {:08X} (branch to {:08X})", address,
               furthest_target);
      } else {
        // Function end point.
        LOGPPC("function end {:08X}", address);
        ends_fn = true;
      }
      ends_block = true;
    } else if (code == 0x4E800420) {
      // bctr -- unconditional branch to CTR.
      // This is generally a jump to a function pointer (non-return).
      // This is almost always a jump table.
      TryRecoverJumpTableAt(function, start_address, address, &furthest_target);
      if (furthest_target > address) {
        // Remaining targets within function, not end.
        LOGPPC("ignoring bctr {:08X} (branch to {:08X})", address,
               furthest_target);
      } else {
        // Function end point.
        LOGPPC("function end {:08X}", address);
        ends_fn = true;
      }
      ends_block = true;
    } else if (opcode == PPCOpcode::bx) {
      // b/ba/bl/bla
      uint32_t target = d.I.ADDR();
      if (d.I.LK()) {
        LOGPPC("bl {:08X} -> {:08X}", address, target);
        // Queue call target if needed.
        // GetOrInsertFunction(target);
      } else {
        LOGPPC("b {:08X} -> {:08X}", address, target);

        // If the target is back into the function and there's no further target
        // we are at the end of a function.
        // (Indirect branches may still go beyond, but no way of knowing).
        if (target >= start_address && target < address &&
            furthest_target <= address) {
          LOGPPC("function end {:08X} (back b)", address);
          ends_fn = true;
        }

        // If the target is not a branch and it goes to before the current
        // address it's definitely a tail call.
        if (!ends_fn && target < start_address && furthest_target <= address) {
          LOGPPC("function end {:08X} (back b before addr)", address);
          ends_fn = true;
        }

        // If the target is a __restgprlr_* method it's the end of a function.
        // Note that sometimes functions stick this in a basic block *inside*
        // of the function somewhere, so ensure we don't have any branches over
        // it.
        if (!ends_fn && furthest_target <= address && IsRestGprLr(target)) {
          LOGPPC("function end {:08X} (__restgprlr_*)", address);
          ends_fn = true;
        }

        // Heuristic: if there's an unconditional branch in the first block of
        // the function it's likely a thunk.
        // Ex:
        //   li        r3, 0
        //   b         KeBugCheck
        // This check may hit on functions that jump over data code, so only
        // trigger this check in leaf functions (no mfspr lr/prolog).
        if (!ends_fn && !starts_with_mfspr_lr && blocks_found == 1) {
          LOGPPC("HEURISTIC: ending at simple leaf thunk {:08X}", address);
          ends_fn = true;
        }

        // Heuristic: if this is an unconditional branch at the end of the
        // function (nothing jumps over us) and we are jumping forward there's
        // a good chance it's a tail call.
        // This may not be true if the code is jumping over data/etc.
        // TODO(benvanik): figure out how to do this reliably. This check as is
        // is too aggressive and turns a lot of valid branches into tail calls.
        // It seems like a lot of functions end up with some prologue bit then
        // jump deep inside only to jump back towards the top soon after. May
        // need something more complex than just a simple 1-pass system to
        // detect these, unless more signals can be found.
        /*
        if (!ends_fn &&
            target > addr &&
            furthest_target < addr) {
          LOGPPC("HEURISTIC: ending at tail call branch %.8X", addr);
          ends_fn = true;
        }
        */

        if (!ends_fn && !IsRestGprLr(target)) {
          furthest_target = std::max(furthest_target, target);

          // TODO(benvanik): perhaps queue up for a speculative check? I think
          //     we are running over tail-call functions here that branch to
          //     somewhere else.
          // GetOrInsertFunction(target);
        }
      }
      ends_block = true;
    } else if (opcode == PPCOpcode::bcx) {
      // bc/bca/bcl/bcla
      uint32_t target = d.B.ADDR();
      if (d.B.LK()) {
        LOGPPC("bcl {:08X} -> {:08X}", address, target);

        // Queue call target if needed.
        // TODO(benvanik): see if this is correct - not sure anyone makes
        //     function calls with bcl.
        // GetOrInsertFunction(target);
      } else {
        LOGPPC("bc {:08X} -> {:08X}", address, target);

        // TODO(benvanik): GetOrInsertFunction? it's likely a BB

        if (!IsRestGprLr(target)) {
          furthest_target = std::max(furthest_target, target);
        }
      }
      ends_block = true;
    } else if (opcode == PPCOpcode::bclrx) {
      // bclr/bclrl
      if (d.XL.LK()) {
        LOGPPC("bclrl {:08X}", address);
      } else {
        LOGPPC("bclr {:08X}", address);
      }
      ends_block = true;
    } else if (opcode == PPCOpcode::bcctrx) {
      // bcctr/bcctrl
      if (d.XL.LK()) {
        LOGPPC("bcctrl {:08X}", address);
      } else {
        LOGPPC("bcctr {:08X}", address);
      }
      ends_block = true;
    }

    if (ends_block) {
      in_block = false;
    }
    if (ends_fn) {
      break;
    }

    address += 4;
  }

  if (!last_address) {
    LOGPPC("function {:08X} has no instructions", start_address);
    return false;
  }

  if (end_address && last_address < end_address) {
    // Ran under the expected value - since we probably got the initial bounds
    // from someplace valid (like method hints) this may indicate an error.
    // It's also possible that we guessed in hole-filling and there's another
    // function below this one.
    LOGPPC("Function ran under: {:08X}-{:08X} ended at {:08X}", start_address,
           end_address, last_address);
  }
  function->set_end_address(last_address);

  // If there's spare bits at the end, split the function.
  // TODO(benvanik): splitting?

  // TODO(benvanik): find and record stack information
  // - look for __savegprlr_* and __restgprlr_*
  // - if present, flag function as needing a stack
  // - record prolog/epilog lengths/stack size/etc

  if (debug_info) {
    debug_info->set_address_reference_count(address_reference_count);
    debug_info->set_instruction_result_count(instruction_result_count);
  }

  LOGPPC("Finished analyzing {:08X}", start_address);
  return true;
}

std::vector<BlockInfo> PPCScanner::FindBlocks(GuestFunction* function) {
  Module* module = function->module();

  std::map<uint32_t, BlockInfo> block_map;

  uint32_t start_address = function->address();
  uint32_t end_address = function->end_address();
  bool in_block = false;
  uint32_t block_start = 0;
  for (uint32_t address = start_address; address <= end_address; address += 4) {
    uint32_t code = xe::load_and_swap<uint32_t>(module->TranslateCode(address));
    if (!code) {
      continue;
    }
    auto opcode = xe::cpu::ppc::LookupOpcode(code);

    if (!in_block) {
      in_block = true;
      block_start = address;
    }

    bool ends_block = false;
    if (code == 0x4E800020) {
      // blr -- unconditional branch to LR.
      ends_block = true;
    } else if (code == 0x4E800420) {
      // bctr -- unconditional branch to CTR.
      // This is almost always a jump table.
      // TODO(benvanik): decode jump tables.
      ends_block = true;
    } else if (opcode == PPCOpcode::bx) {
      // b/ba/bl/bla
      // uint32_t target =
      //    (uint32_t)XEEXTS26(i.I.LI << 2) + (i.I.AA ? 0 : (int32_t)address);
      ends_block = true;
    } else if (opcode == PPCOpcode::bcx) {
      // bc/bca/bcl/bcla
      // uint32_t target =
      //    (uint32_t)XEEXTS16(i.B.BD << 2) + (i.B.AA ? 0 : (int32_t)address);
      ends_block = true;
    } else if (opcode == PPCOpcode::bclrx) {
      // bclr/bclrl
      ends_block = true;
    } else if (opcode == PPCOpcode::bcctrx) {
      // bcctr/bcctrl
      ends_block = true;
    }

    if (ends_block) {
      in_block = false;
      block_map[block_start] = {
          block_start,
          address,
      };
    }
  }

  if (in_block) {
    block_map[block_start] = {
        block_start,
        end_address,
    };
  }

  std::vector<BlockInfo> blocks;
  for (auto it = block_map.begin(); it != block_map.end(); ++it) {
    blocks.push_back(it->second);
  }
  return blocks;
}

}  // namespace ppc
}  // namespace cpu
}  // namespace xe
