/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/base/logging.h"
#include "xenia/kernel/kernel_state.h"
#include "xenia/kernel/user_module.h"
#include "xenia/kernel/util/shim_utils.h"
#include "xenia/kernel/xam/xam_private.h"
#include "xenia/kernel/xboxkrnl/xboxkrnl_error.h"
#include "xenia/kernel/xboxkrnl/xboxkrnl_threading.h"
#include "xenia/kernel/xthread.h"
#include "xenia/xbox.h"

#include "third_party/fmt/include/fmt/format.h"

namespace xe {
namespace kernel {
namespace xam {

struct XTASK_MESSAGE {
  be<uint32_t> unknown_00;
  be<uint32_t> unknown_04;
  be<uint32_t> unknown_08;
  be<uint32_t> callback_arg_ptr;
  be<uint32_t> event_handle;
  be<uint32_t> unknown_14;
  be<uint32_t> task_handle;
};

struct XAM_TASK_ARGS {
  be<uint32_t> flags;
  be<uint32_t> value2;
  // i think there might be another value here, it might be padding
};
static_assert_size(XTASK_MESSAGE, 0x1C);

dword_result_t XamTaskSchedule_entry(lpvoid_t callback,
                                     pointer_t<XTASK_MESSAGE> message,
                                     dword_t optional_ptr, lpdword_t handle_ptr,
                                     const ppc_context_t& ctx) {
  // TODO(gibbed): figure out what this is for
  *handle_ptr = 12345;

  uint32_t task_flags = 0;
  if (optional_ptr) {
    auto option = ctx->TranslateVirtual<XAM_TASK_ARGS*>(optional_ptr);

    task_flags = option->flags;
    auto v2 = option->value2;  // typically 0?

    XELOGI("Got xam task args: v1 = {:08X}, v2 = {:08X}", task_flags, v2.get());
  }

  auto exe_module = kernel_state()->GetExecutableModule();
  uint32_t stack_size = exe_module ? exe_module->stack_size() : 0x4000;

  // Stack must be aligned to 16kb pages
  stack_size = std::max((uint32_t)0x4000, ((stack_size + 0xFFF) & 0xFFFFF000));

  uint32_t system_flags = 0;
  if (exe_module) {
    exe_module->GetOptHeader(XEX_HEADER_SYSTEM_FLAGS, &system_flags);
  }
  const bool foreground_task =
      ((task_flags & 2u) != 0) ||
      ((system_flags & XEX_SYSTEM_FOREGROUND_TASKS) != 0);

  // Schedule XAM tasks on CPU 5 (AffinityCpu5 = 0x20000000). Create suspended
  // when promoting to title priority so the thread does not start at system
  // priority before its priority class and base priority are updated.
  constexpr uint32_t kXamTaskCpu5Flag = 0x20000000u;
  const uint32_t creation_flags =
      kXamTaskCpu5Flag | (foreground_task ? X_CREATE_SUSPENDED : 0u);

  auto thread = object_ref<XThread>(new XThread(
      kernel_state(), stack_size, 0, callback, message.guest_address(),
      creation_flags, true, false, kernel_state()->GetSystemProcess()));

  X_STATUS result = thread->Create();

  if (XFAILED(result)) {
    // Failed!
    XELOGE("XAM task creation failed: {:08X}", result);
    return result;
  }

  if (foreground_task) {
    auto* title_process =
        kernel_state()->memory()->TranslateVirtual<X_KPROCESS*>(
            kernel_state()->GetTitleProcess());
    auto* kthread = thread->guest_object<X_KTHREAD>();
    kthread->process_priority_class = title_process->process_priority_class;
    kthread->base_priority_copy = title_process->default_thread_priority;
    kthread->max_dynamic_priority = title_process->max_dynamic_priority;
    kthread->creation_flags = kXamTaskCpu5Flag;
    thread->SetBasePriority(0);
    thread->Resume();
  }

  XELOGD("XAM task ({:08X}) scheduled asynchronously",
         callback.guest_address());

  return X_STATUS_SUCCESS;
}
DECLARE_XAM_EXPORT2(XamTaskSchedule, kNone, kImplemented, kSketchy);

dword_result_t XamTaskCloseHandle_entry(dword_t obj_handle) {
  const X_STATUS error_code = xboxkrnl::NtClose(obj_handle);

  if (XFAILED(error_code)) {
    const uint32_t result = xboxkrnl::xeRtlNtStatusToDosError(error_code);
    XThread::SetLastError(result);
    return false;
  }

  return true;
}
DECLARE_XAM_EXPORT1(XamTaskCloseHandle, kNone, kImplemented);

dword_result_t XamTaskShouldExit_entry() { return 0; }
DECLARE_XAM_EXPORT1(XamTaskShouldExit, kNone, kStub);

}  // namespace xam
}  // namespace kernel
}  // namespace xe

DECLARE_XAM_EMPTY_REGISTER_EXPORTS(Task);
