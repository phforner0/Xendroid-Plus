/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2013 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_APU_AUDIO_DRIVER_H_
#define XENIA_APU_AUDIO_DRIVER_H_

#include "xenia/memory.h"
#include "xenia/xbox.h"

namespace xe {
namespace apu {

class AudioDriver {
 public:
  static constexpr uint32_t kFrameFrequencyDefault = 48000;
  static constexpr uint32_t kFrameChannelsDefault = 6;
  static constexpr uint32_t kChannelSamplesDefault = 256;
  static constexpr uint32_t kFrameSamplesMax =
      kFrameChannelsDefault * kChannelSamplesDefault;
  static constexpr uint32_t kFrameSizeMax = sizeof(float) * kFrameSamplesMax;

  virtual ~AudioDriver();

  virtual bool Initialize() = 0;
  virtual void Shutdown() = 0;

  virtual void SubmitFrame(float* samples) = 0;
  virtual void Pause() = 0;
  virtual void Resume() = 0;
  virtual void SetVolume(float volume) = 0;
  // Frames submitted but not yet played. Drivers that cannot report it leave
  // the default, which keeps callers on their paced path.
  virtual size_t GetQueuedFrameCount() { return 0; }
  // Whether GetQueuedFrameCount is the real depth, not the default 0.
  virtual bool ReportsQueuedFrameCount() const { return false; }
  // Hint from AudioSystem for adaptive queue depth target in blocks so
  // driver-side dynamic rate catch-up does not drain ahead of it.
  virtual void SetTargetQueueDepth(size_t /*blocks*/) {}
};

}  // namespace apu
}  // namespace xe

#endif  // XENIA_APU_AUDIO_DRIVER_H_
